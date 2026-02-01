//! Search command - query an index

use clap::Args;
use tracing::info;

use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{find_index, IndexMeta, IndexSearcher, MetadataFilter, RecomputeSearcher, SearchOptions, SearchResult};

#[derive(Args)]
pub struct SearchArgs {
    /// Index name to search
    pub index_name: String,

    /// Search query
    pub query: String,

    /// Number of results to return
    #[arg(long, default_value = "5")]
    pub top_k: usize,

    /// Search complexity (higher = more accurate but slower)
    #[arg(long, default_value = "64")]
    pub complexity: usize,

    /// Show file paths in results
    #[arg(long)]
    pub show_metadata: bool,

    /// Filter results by metadata (e.g., "source:*.rs" or "type=code")
    #[arg(long, short = 'f')]
    pub filter: Option<String>,

    /// Enable hybrid search (vector + BM25)
    #[arg(long)]
    pub hybrid: bool,

    /// Weight for vector scores in hybrid mode (0.0-1.0, default 0.7)
    #[arg(long, default_value = "0.7")]
    pub hybrid_alpha: f32,

    /// Output format (text, json)
    #[arg(long, default_value = "text", value_parser = ["text", "json"])]
    pub format: String,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,

    /// Query prompt template prefix for asymmetric embedding models
    /// (e.g., "query: " for E5 models, or custom prefix for Instructor models)
    #[arg(long)]
    pub query_prompt_template: Option<String>,
}

pub async fn run(args: SearchArgs, _verbose: bool) -> anyhow::Result<()> {
    // Find index
    let index_dir = find_index(&args.index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    // Load metadata
    let meta = IndexMeta::load(&meta_path)?;

    // Check if index is pruned (needs recomputation)
    let is_pruned = meta.is_pruned;

    info!(
        "Searching index '{}' ({} passages, {} dims){}",
        args.index_name, meta.passage_count, meta.dimensions,
        if is_pruned { " [recompute mode]" } else { "" }
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
        "gemini" => EmbeddingMode::Gemini {
            api_key: std::env::var("GOOGLE_API_KEY").ok(),
        },
        _ => anyhow::bail!("Unknown embedding mode in index: {}", meta.embedding_mode),
    };

    let embedding_provider = EmbeddingProvider::new(
        meta.embedding_model.clone(),
        embedding_mode,
    ).await?;

    // Compute query embedding (with optional template for asymmetric models)
    let query_template = args.query_prompt_template.as_deref().unwrap_or("");
    let query_embedding = embedding_provider
        .embed_with_template(&[&args.query], query_template)
        .await?;
    let query_embedding = &query_embedding[0];

    // Parse filter
    let filter = if let Some(filter_str) = &args.filter {
        if let Some(f) = MetadataFilter::parse(filter_str) {
            Some(f)
        } else {
            anyhow::bail!("Invalid filter syntax: {}", filter_str);
        }
    } else {
        None
    };

    // Search - use recompute mode if index is pruned
    let results: Vec<SearchResult> = if is_pruned {
        if args.hybrid {
            info!("Note: Hybrid search is not supported in recompute mode, using vector search only");
        }

        let searcher = RecomputeSearcher::load(&index_path, meta.dimensions)?;
        searcher.search(
            query_embedding,
            &embedding_provider,
            args.top_k,
            filter.as_ref(),
        ).await?
    } else {
        // Normal search with vector index
        let searcher = IndexSearcher::load(&index_path, &meta)?;

        let mut opts = SearchOptions::new(args.top_k, args.complexity);

        if let Some(f) = filter {
            opts = opts.with_filter(f);
        }

        if args.hybrid {
            opts = opts.with_hybrid(args.query.clone(), args.hybrid_alpha);
        }

        searcher.search_with_options(query_embedding, &opts)?
    };

    // Output results
    if args.format == "json" {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    } else {
        println!("\nSearch results for '{}' (top {}):\n", args.query, results.len());

        for (i, result) in results.iter().enumerate() {
            println!("{}. Score: {:.4}", i + 1, result.score);

            if args.show_metadata {
                if let Some(source) = result.metadata.get("source") {
                    println!("   Source: {}", source);
                }
                // Show other metadata fields
                if let Some(obj) = result.metadata.as_object() {
                    for (key, value) in obj {
                        if key != "source" {
                            println!("   {}: {}", key, value);
                        }
                    }
                }
            }

            // Truncate text for display
            let display_text = if result.text.len() > 200 {
                format!("{}...", &result.text[..200])
            } else {
                result.text.clone()
            };
            println!("   {}", display_text);
            println!();
        }
    }

    Ok(())
}

