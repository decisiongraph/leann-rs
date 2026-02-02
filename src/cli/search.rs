//! Search command - query an index

use clap::Args;
use tracing::info;

use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{expand_from_passages, find_index, IndexMeta, IndexSearcher, MetadataFilter, RecomputeSearcher, SearchOptions, SearchResult, should_expand};

#[derive(Args)]
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Index name to search (defaults to current directory name)
    #[arg(short, long)]
    pub index: Option<String>,

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
    /// Use "auto" to automatically enable for short queries (1-3 words)
    #[arg(long)]
    pub hybrid: bool,

    /// Auto-enable hybrid search for short queries (1-3 words)
    #[arg(long, default_value = "true")]
    pub auto_hybrid: bool,

    /// Expand short queries with related terms for better recall
    #[arg(long, default_value = "true")]
    pub expand: bool,

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
    // Default to current directory name if no index specified
    let index_name = args.index.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "index".to_string())
    });

    // Find index
    let index_dir = find_index(&index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    // Load metadata
    let meta = IndexMeta::load(&meta_path)?;

    // Check if index is pruned (needs recomputation)
    let is_pruned = meta.is_pruned;

    info!(
        "Searching index '{}' ({} passages, {} dims){}",
        index_name, meta.passage_count, meta.dimensions,
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

    // Get query template from CLI args, metadata, or model defaults
    let query_template = args.query_prompt_template.clone().unwrap_or_else(|| {
        // Try to get from index metadata
        if let Some(opts) = &meta.embedding_options {
            if let Some(template) = opts.get("query_prompt_template").and_then(|v| v.as_str()) {
                return template.to_string();
            }
        }
        // Fall back to model defaults
        crate::embedding::get_model_config(&meta.embedding_model).query_prefix.to_string()
    });

    if !query_template.is_empty() {
        tracing::debug!("Using query prefix: {:?}", query_template);
    }

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

    // Determine if hybrid search should be used
    let word_count = args.query.split_whitespace().count();
    let use_hybrid = args.hybrid || (args.auto_hybrid && word_count <= 3);

    // Search - use recompute mode if index is pruned
    let results: Vec<SearchResult> = if is_pruned {
        if use_hybrid {
            info!("Note: Hybrid search is not supported in recompute mode, using vector search only");
        }

        // No expansion in recompute mode
        let query_embedding = embedding_provider
            .embed_with_template(&[&args.query], &query_template)
            .await?;

        let searcher = RecomputeSearcher::load(&index_path, meta.dimensions)?;
        searcher.search(
            &query_embedding[0],
            &embedding_provider,
            args.top_k,
            filter.as_ref(),
        ).await?
    } else {
        // Normal search with vector index
        let searcher = IndexSearcher::load(&index_path, &meta)?;

        // Expand query using BM25 matches if enabled
        let search_query = if args.expand && should_expand(&args.query) {
            let bm25_texts = searcher.bm25_search(&args.query, 5)?;
            if !bm25_texts.is_empty() {
                let text_refs: Vec<&str> = bm25_texts.iter().map(|s| s.as_str()).collect();
                let expanded = expand_from_passages(&args.query, &text_refs, 5);
                if expanded != args.query {
                    info!("Expanded query: '{}' → '{}'", args.query, expanded);
                }
                expanded
            } else {
                args.query.clone()
            }
        } else {
            args.query.clone()
        };

        // Compute query embedding
        let query_embedding = embedding_provider
            .embed_with_template(&[&search_query], &query_template)
            .await?;

        let mut opts = SearchOptions::new(args.top_k, args.complexity);

        if let Some(f) = filter {
            opts = opts.with_filter(f);
        }

        if use_hybrid {
            if word_count <= 3 && !args.hybrid {
                tracing::debug!("Auto-enabling hybrid search for short query ({} words)", word_count);
            }
            opts = opts.with_hybrid(search_query.clone(), args.hybrid_alpha);
        }

        searcher.search_with_options(&query_embedding[0], &opts)?
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

            // Truncate text for display (respecting UTF-8 boundaries)
            let display_text = if result.text.len() > 200 {
                let mut end = 200;
                while end > 0 && !result.text.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &result.text[..end])
            } else {
                result.text.clone()
            };
            println!("   {}", display_text);
            println!();
        }
    }

    Ok(())
}

