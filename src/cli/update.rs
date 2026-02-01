//! Update command - add passages to an existing index

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use crate::backend::{BackendBuilder, BackendType};
use crate::chunker::{Chunk, ChunkingStrategy};
use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{IndexMeta, PassageStore, Passage};

use super::build::load_documents;

#[derive(Args)]
pub struct UpdateArgs {
    /// Index name to update
    pub index_name: String,

    /// Document directories and/or files to add
    #[arg(long, required = true)]
    pub docs: Vec<PathBuf>,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,

    /// Document chunk size in tokens
    #[arg(long, default_value = "256")]
    pub doc_chunk_size: usize,

    /// Document chunk overlap in tokens
    #[arg(long, default_value = "128")]
    pub doc_chunk_overlap: usize,

    /// File types to include (comma-separated)
    #[arg(long)]
    pub file_types: Option<String>,

    /// Include hidden files
    #[arg(long)]
    pub include_hidden: bool,

    /// Chunking strategy: simple, ast, or auto (default: auto)
    #[arg(long, default_value = "auto", value_parser = ["simple", "ast", "auto"])]
    pub chunking_strategy: String,
}

pub async fn run(args: UpdateArgs, _verbose: bool) -> anyhow::Result<()> {
    info!("Updating index '{}'", args.index_name);

    // Find index
    let index_dir = find_index(&args.index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    // Load metadata
    let mut meta = IndexMeta::load(&meta_path)?;

    // Check backend supports updates
    let backend_type = match meta.backend_name.as_str() {
        "hnsw" => BackendType::Hnsw,
        "diskann" => anyhow::bail!(
            "DiskANN backend does not support incremental updates. \
            Use 'leann build --force' to rebuild the entire index."
        ),
        _ => anyhow::bail!("Unknown backend: {}", meta.backend_name),
    };

    info!(
        "Current index: {} passages, {} dimensions",
        meta.passage_count, meta.dimensions
    );

    // Create embedding provider from metadata
    let embedding_mode = match meta.embedding_mode.as_str() {
        "openai" => EmbeddingMode::OpenAI {
            api_key: args.embedding_api_key.clone(),
            base_url: args.embedding_api_base.clone(),
        },
        "ollama" => EmbeddingMode::Ollama {
            host: args.embedding_host.clone(),
        },
        _ => anyhow::bail!("Unknown embedding mode in index: {}", meta.embedding_mode),
    };

    let embedding_provider = EmbeddingProvider::new(
        meta.embedding_model.clone(),
        embedding_mode,
    ).await?;

    // Verify dimensions match
    if embedding_provider.dimensions() != meta.dimensions {
        anyhow::bail!(
            "Embedding dimension mismatch: index has {}, provider has {}",
            meta.dimensions,
            embedding_provider.dimensions()
        );
    }

    // Load new documents
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    progress.set_message("Loading new documents...");

    let file_types: Option<Vec<String>> = args.file_types.map(|ft| {
        ft.split(',')
            .map(|s| s.trim().to_string())
            .collect()
    });

    let chunking_strategy: ChunkingStrategy = args.chunking_strategy.parse()
        .unwrap_or(ChunkingStrategy::Auto);

    let chunks = load_documents(
        &args.docs,
        args.doc_chunk_size,
        args.doc_chunk_overlap,
        file_types.as_deref(),
        args.include_hidden,
        chunking_strategy,
    )?;

    progress.finish_with_message(format!("Loaded {} new chunks", chunks.len()));

    if chunks.is_empty() {
        println!("No new documents found to add");
        return Ok(());
    }

    // Compute embeddings for new chunks
    let progress = ProgressBar::new(chunks.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress.set_message("Computing embeddings...");

    let batch_size = 100;
    let mut all_embeddings = Vec::with_capacity(chunks.len());

    for batch in chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedding_provider.embed(&texts).await?;
        all_embeddings.extend(embeddings);
        progress.inc(batch.len() as u64);
    }

    progress.finish_with_message("Embeddings computed");

    // Open passage store for appending
    let mut passage_writer = PassageStore::open_for_append(&index_path)?;
    let start_id = meta.passage_count;

    // Reassign IDs starting from current count
    let new_chunks: Vec<Chunk> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.id = (start_id + i).to_string();
            c
        })
        .collect();

    // Add passages to store
    let progress = ProgressBar::new(new_chunks.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Adding passages...")
            .unwrap()
            .progress_chars("#>-"),
    );

    for chunk in &new_chunks {
        let passage = Passage {
            id: chunk.id.clone(),
            text: chunk.text.clone(),
            metadata: chunk.metadata.clone(),
        };
        passage_writer.add(&passage)?;
        progress.inc(1);
    }

    passage_writer.finish()?;
    progress.finish_with_message("Passages added");

    // Update IDs file
    let ids_path = index_path.with_extension("ids.txt");
    let mut ids_content = if ids_path.exists() {
        std::fs::read_to_string(&ids_path)?
    } else {
        String::new()
    };

    for chunk in &new_chunks {
        if !ids_content.is_empty() {
            ids_content.push('\n');
        }
        ids_content.push_str(&chunk.id);
    }
    std::fs::write(&ids_path, ids_content)?;

    // Add embeddings to vector index
    let backend = BackendBuilder::new(backend_type);
    backend.add_to_index(
        &all_embeddings,
        &index_path,
        meta.dimensions,
        start_id,
    )?;

    // Update metadata
    meta.passage_count += new_chunks.len();
    meta.save(&meta_path)?;

    println!(
        "Index '{}' updated: {} → {} passages",
        args.index_name,
        start_id,
        meta.passage_count
    );

    Ok(())
}

/// Find an index by name
fn find_index(name: &str) -> anyhow::Result<PathBuf> {
    let local_path = PathBuf::from(".leann").join("indexes").join(name);
    if local_path.exists() {
        return Ok(local_path);
    }

    let abs_path = PathBuf::from(name);
    if abs_path.is_absolute() && abs_path.exists() {
        return Ok(abs_path);
    }

    if let Some(home) = dirs::home_dir() {
        let global_path = home.join(".leann").join("indexes").join(name);
        if global_path.exists() {
            return Ok(global_path);
        }
    }

    anyhow::bail!(
        "Index '{}' not found. Run 'leann list' to see available indexes.",
        name
    )
}
