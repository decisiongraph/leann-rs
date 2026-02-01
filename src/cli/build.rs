//! Build command - index construction from documents

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use crate::backend::BackendType;
use crate::chunker::{Chunk, Chunker, ChunkingStrategy, SmartChunker};
use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{IndexBuilder, IndexMeta};

#[derive(Args)]
pub struct BuildArgs {
    /// Index name (default: current directory name)
    #[arg()]
    pub index_name: Option<String>,

    /// Document directories and/or files
    #[arg(long, default_value = ".")]
    pub docs: Vec<PathBuf>,

    /// Backend to use
    #[arg(long, default_value = "hnsw", value_parser = ["hnsw", "diskann"])]
    pub backend_name: String,

    /// Embedding model name
    #[arg(long, default_value = "text-embedding-3-small")]
    pub embedding_model: String,

    /// Embedding mode
    #[cfg(feature = "local-embeddings")]
    #[arg(long, default_value = "openai", value_parser = ["openai", "ollama", "gemini", "local"])]
    pub embedding_mode: String,

    /// Embedding mode
    #[cfg(not(feature = "local-embeddings"))]
    #[arg(long, default_value = "openai", value_parser = ["openai", "ollama", "gemini"])]
    pub embedding_mode: String,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// API key for embedding service (OpenAI)
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// Google API key for Gemini embeddings
    #[arg(long, env = "GOOGLE_API_KEY")]
    pub google_api_key: Option<String>,

    /// Prompt template prefix for document embeddings
    /// Used for asymmetric models (e.g., "passage: " for E5, BGE models)
    #[arg(long)]
    pub embedding_prompt_template: Option<String>,

    /// Local model path (for local embedding mode)
    #[cfg(feature = "local-embeddings")]
    #[arg(long)]
    pub embedding_model_path: Option<String>,

    /// Force rebuild existing index
    #[arg(short, long)]
    pub force: bool,

    /// Enable recomputation mode (stores embeddings separately for pruning)
    #[arg(long)]
    pub recompute: bool,

    /// Graph degree for HNSW
    #[arg(long, default_value = "32")]
    pub graph_degree: usize,

    /// Build complexity
    #[arg(long, default_value = "64")]
    pub complexity: usize,

    /// Document chunk size in tokens
    #[arg(long, default_value = "256")]
    pub doc_chunk_size: usize,

    /// Document chunk overlap in tokens
    #[arg(long, default_value = "128")]
    pub doc_chunk_overlap: usize,

    /// File types to include (comma-separated, e.g., ".txt,.pdf,.md")
    #[arg(long)]
    pub file_types: Option<String>,

    /// Include hidden files
    #[arg(long)]
    pub include_hidden: bool,

    /// Chunking strategy: simple, ast, or auto (default: auto)
    /// - simple: character-based chunking with word boundary awareness
    /// - ast: AST-aware chunking for code files (functions, classes, etc.)
    /// - auto: automatically selects based on file type
    #[arg(long, default_value = "auto", value_parser = ["simple", "ast", "auto"])]
    pub chunking_strategy: String,
}

pub async fn run(args: BuildArgs, _verbose: bool) -> anyhow::Result<()> {
    let index_name = args.index_name.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "index".to_string())
    });

    info!("Building index '{}'", index_name);

    // Determine index directory
    let index_dir = PathBuf::from(".leann").join("indexes").join(&index_name);

    if index_dir.exists() && !args.force {
        anyhow::bail!(
            "Index '{}' already exists. Use --force to rebuild.",
            index_name
        );
    }

    // Create index directory
    std::fs::create_dir_all(&index_dir)?;

    // Parse backend type
    let backend_type = match args.backend_name.as_str() {
        "hnsw" => BackendType::Hnsw,
        "diskann" => BackendType::DiskAnn,
        _ => anyhow::bail!("Unknown backend: {}", args.backend_name),
    };

    // Parse embedding mode
    let embedding_mode = match args.embedding_mode.as_str() {
        "openai" => EmbeddingMode::OpenAI {
            api_key: args.embedding_api_key.clone(),
            base_url: args.embedding_api_base.clone(),
        },
        "ollama" => EmbeddingMode::Ollama {
            host: args.embedding_host.clone(),
        },
        "gemini" => EmbeddingMode::Gemini {
            api_key: args.google_api_key.clone(),
        },
        #[cfg(feature = "local-embeddings")]
        "local" => EmbeddingMode::Local {
            model_path: args.embedding_model_path.clone(),
        },
        _ => anyhow::bail!("Unknown embedding mode: {}", args.embedding_mode),
    };

    // Create embedding provider
    let embedding_provider = EmbeddingProvider::new(
        args.embedding_model.clone(),
        embedding_mode.clone(),
    ).await?;

    // Get embedding dimensions
    let dimensions = embedding_provider.dimensions();
    info!("Embedding dimensions: {}", dimensions);

    // Load documents
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    progress.set_message("Loading documents...");

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

    progress.finish_with_message(format!("Loaded {} chunks", chunks.len()));

    if chunks.is_empty() {
        anyhow::bail!("No documents found to index");
    }

    // Build index
    let mut builder = IndexBuilder::new(
        backend_type,
        dimensions,
        args.graph_degree,
        args.complexity,
    ).with_recompute_mode(args.recompute);

    let progress = ProgressBar::new(chunks.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress.set_message("Computing embeddings...");

    // Process chunks in batches for efficiency
    let batch_size = 100;
    let mut all_embeddings = Vec::with_capacity(chunks.len());
    let embed_template = args.embedding_prompt_template.as_deref().unwrap_or("");

    for batch in chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedding_provider
            .embed_with_template(&texts, embed_template)
            .await?;
        all_embeddings.extend(embeddings);
        progress.inc(batch.len() as u64);
    }

    progress.finish_with_message("Embeddings computed");

    // Add chunks to builder
    let progress = ProgressBar::new(chunks.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Building index...")
            .unwrap()
            .progress_chars("#>-"),
    );

    for (i, chunk) in chunks.iter().enumerate() {
        builder.add_passage(
            &chunk.id,
            &chunk.text,
            &all_embeddings[i],
            chunk.metadata.clone(),
        )?;
        progress.inc(1);
    }

    progress.finish_with_message("Index built");

    // Save index
    let index_path = index_dir.join("documents.leann");
    builder.build(&index_path)?;

    // Save metadata
    let meta = IndexMeta {
        version: "1.0".to_string(),
        backend_name: args.backend_name,
        embedding_model: args.embedding_model,
        embedding_mode: args.embedding_mode,
        dimensions,
        passage_count: chunks.len(),
        backend_kwargs: None,
        embedding_options: None,
        is_recompute: args.recompute,
        is_pruned: false,
    };
    meta.save(&index_dir.join("documents.leann.meta.json"))?;

    println!("Index '{}' built successfully at {:?}", index_name, index_dir);
    println!("  Passages: {}", chunks.len());
    println!("  Dimensions: {}", dimensions);
    if args.recompute {
        println!("  Recompute mode: enabled (run 'leann prune {}' to save space)", index_name);
    }

    Ok(())
}

/// Load documents from paths and chunk them
pub fn load_documents(
    paths: &[PathBuf],
    chunk_size: usize,
    chunk_overlap: usize,
    file_types: Option<&[String]>,
    include_hidden: bool,
    chunking_strategy: ChunkingStrategy,
) -> anyhow::Result<Vec<Chunk>> {
    use ignore::WalkBuilder;

    let chunker = SmartChunker::new(chunking_strategy, chunk_size, chunk_overlap);
    let mut chunks = Vec::new();
    let mut chunk_id = 0u64;

    // Default file types
    #[cfg(feature = "pdf")]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
        ".pdf",
    ];
    #[cfg(not(feature = "pdf"))]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
    ];

    let allowed_extensions: Vec<&str> = file_types
        .map(|ft| ft.iter().map(|s| s.as_str()).collect())
        .unwrap_or(default_types);

    for path in paths {
        if path.is_file() {
            // Single file
            if let Some(ext) = path.extension() {
                let ext_str = format!(".{}", ext.to_string_lossy());
                if allowed_extensions.iter().any(|e| *e == ext_str) {
                    if let Some(content) = load_file_content(path) {
                        let file_chunks = chunker.chunk(&content, path, &mut chunk_id);
                        chunks.extend(file_chunks);
                    }
                }
            }
        } else if path.is_dir() {
            // Directory - walk with gitignore support
            let walker = WalkBuilder::new(path)
                .hidden(!include_hidden)
                .git_ignore(true)
                .git_global(true)
                .build();

            for entry in walker.flatten() {
                let entry_path = entry.path();
                if entry_path.is_file() {
                    if let Some(ext) = entry_path.extension() {
                        let ext_str = format!(".{}", ext.to_string_lossy());
                        if allowed_extensions.iter().any(|e| *e == ext_str) {
                            if let Some(content) = load_file_content(entry_path) {
                                let file_chunks = chunker.chunk(&content, entry_path, &mut chunk_id);
                                chunks.extend(file_chunks);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(chunks)
}

/// Load file content, handling different file types
fn load_file_content(path: &std::path::Path) -> Option<String> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        #[cfg(feature = "pdf")]
        "pdf" => {
            match pdf_extract::extract_text(path) {
                Ok(text) => {
                    let text = text.trim();
                    if text.is_empty() {
                        tracing::warn!("PDF {} contains no extractable text", path.display());
                        None
                    } else {
                        Some(text.to_string())
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to extract text from {}: {}", path.display(), e);
                    None
                }
            }
        }
        _ => {
            // Regular text file
            std::fs::read_to_string(path).ok()
        }
    }
}
