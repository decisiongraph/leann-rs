//! Serve command - HTTP API server

use clap::Args;

#[derive(Args)]
pub struct ServeArgs {
    /// Index name to serve
    pub index_name: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Enable CORS for all origins
    #[arg(long)]
    pub cors: bool,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,
}

#[cfg(feature = "server")]
pub async fn run(args: ServeArgs, _verbose: bool) -> anyhow::Result<()> {
    use std::sync::Arc;

    use axum::{
        routing::{get, post},
        Router,
    };
    use tower_http::cors::{Any, CorsLayer};
    use tokio::sync::RwLock;
    use tracing::info;

    use crate::embedding::{EmbeddingMode, EmbeddingProvider};
    use crate::index::{find_index, IndexMeta, IndexSearcher};

    // Find and load index
    let index_dir = find_index(&args.index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    let meta = IndexMeta::load(&meta_path)?;

    info!(
        "Serving index '{}' ({} passages, {} dims)",
        args.index_name, meta.passage_count, meta.dimensions
    );

    // Create embedding provider
    let embedding_mode = match meta.embedding_mode.as_str() {
        "openai" => EmbeddingMode::OpenAI {
            api_key: args.embedding_api_key.clone(),
            base_url: None,
        },
        "ollama" => EmbeddingMode::Ollama {
            host: args.embedding_host.clone(),
        },
        "gemini" => EmbeddingMode::Gemini {
            api_key: std::env::var("GOOGLE_API_KEY").ok(),
        },
        _ => anyhow::bail!("Unknown embedding mode: {}", meta.embedding_mode),
    };

    let embedding_provider = EmbeddingProvider::new(
        meta.embedding_model.clone(),
        embedding_mode,
    ).await?;

    // Load index
    let searcher = IndexSearcher::load(&index_path, &meta)?;

    // Shared state
    let state = Arc::new(AppState {
        embedding_provider: RwLock::new(embedding_provider),
        searcher: RwLock::new(searcher),
        index_name: args.index_name.clone(),
        meta,
    });

    // Build router
    let mut app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/indexes", get(list_indexes))
        .route("/search", post(search))
        .route("/info", get(info_handler))
        .with_state(state);

    if args.cors {
        app = app.layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any));
    }

    let addr = format!("{}:{}", args.host, args.port);
    println!("LEANN server listening on http://{}", addr);
    println!("  GET  /indexes - List available indexes");
    println!("  POST /search  - Search the index");
    println!("  GET  /info    - Get index information");
    println!("  GET  /health  - Health check");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(feature = "server")]
struct AppState {
    embedding_provider: tokio::sync::RwLock<crate::embedding::EmbeddingProvider>,
    searcher: tokio::sync::RwLock<crate::index::IndexSearcher>,
    index_name: String,
    meta: crate::index::IndexMeta,
}

#[cfg(feature = "server")]
async fn root() -> &'static str {
    "LEANN API Server\n\nEndpoints:\n  POST /search - Search the index\n  GET  /info   - Get index information\n  GET  /health - Health check\n"
}

#[cfg(feature = "server")]
async fn health() -> &'static str {
    "ok"
}

#[cfg(feature = "server")]
async fn info_handler(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<AppState>>,
) -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({
        "index_name": state.index_name,
        "passage_count": state.meta.passage_count,
        "dimensions": state.meta.dimensions,
        "embedding_model": state.meta.embedding_model,
        "backend": state.meta.backend_name,
    }))
}

#[cfg(feature = "server")]
#[derive(serde::Serialize)]
struct IndexInfo {
    name: String,
    status: String,
    size_mb: f64,
    passage_count: Option<usize>,
    backend: Option<String>,
}

#[cfg(feature = "server")]
async fn list_indexes() -> axum::response::Json<Vec<IndexInfo>> {
    use std::path::PathBuf;

    let mut indexes = Vec::new();

    // Check local .leann/indexes
    let local_path = PathBuf::from(".leann").join("indexes");
    if local_path.exists() {
        if let Ok(entries) = std::fs::read_dir(&local_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let meta_path = entry.path().join("documents.leann.meta.json");

                    let (status, passage_count, backend) = if meta_path.exists() {
                        if let Ok(content) = std::fs::read_to_string(&meta_path) {
                            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
                                let pc = meta.get("passage_count").and_then(|v| v.as_u64()).map(|v| v as usize);
                                let be = meta.get("backend_name").and_then(|v| v.as_str()).map(|s| s.to_string());
                                ("ready".to_string(), pc, be)
                            } else {
                                ("invalid".to_string(), None, None)
                            }
                        } else {
                            ("error".to_string(), None, None)
                        }
                    } else {
                        ("incomplete".to_string(), None, None)
                    };

                    // Calculate directory size
                    let size_mb = dir_size(&entry.path()).unwrap_or(0) as f64 / (1024.0 * 1024.0);

                    indexes.push(IndexInfo {
                        name,
                        status,
                        size_mb,
                        passage_count,
                        backend,
                    });
                }
            }
        }
    }

    axum::response::Json(indexes)
}

#[cfg(feature = "server")]
fn dir_size(path: &std::path::PathBuf) -> std::io::Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                size += entry.metadata()?.len();
            } else if path.is_dir() {
                size += dir_size(&path)?;
            }
        }
    }
    Ok(size)
}

#[cfg(feature = "server")]
#[derive(serde::Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    hybrid: bool,
    #[serde(default = "default_alpha")]
    hybrid_alpha: f32,
}

#[cfg(feature = "server")]
fn default_top_k() -> usize { 5 }

#[cfg(feature = "server")]
fn default_alpha() -> f32 { 0.7 }

#[cfg(feature = "server")]
#[derive(serde::Serialize)]
struct SearchResponse {
    results: Vec<SearchResultJson>,
    query: String,
    took_ms: u64,
}

#[cfg(feature = "server")]
#[derive(serde::Serialize)]
struct SearchResultJson {
    id: String,
    score: f32,
    text: String,
    metadata: serde_json::Value,
}

#[cfg(feature = "server")]
async fn search(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<AppState>>,
    axum::Json(req): axum::Json<SearchRequest>,
) -> Result<axum::response::Json<SearchResponse>, (axum::http::StatusCode, String)> {
    use crate::index::{MetadataFilter, SearchOptions};

    let start = std::time::Instant::now();

    // Compute embedding
    let embedding_provider = state.embedding_provider.read().await;
    let query_embedding = embedding_provider
        .embed(&[&req.query])
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Build search options
    let mut opts = SearchOptions::new(req.top_k, 64);

    if let Some(filter_str) = &req.filter {
        if let Some(filter) = MetadataFilter::parse(filter_str) {
            opts = opts.with_filter(filter);
        }
    }

    if req.hybrid {
        opts = opts.with_hybrid(req.query.clone(), req.hybrid_alpha);
    }

    // Search
    let searcher = state.searcher.read().await;
    let results = searcher
        .search_with_options(&query_embedding[0], &opts)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let took_ms = start.elapsed().as_millis() as u64;

    let response = SearchResponse {
        results: results
            .into_iter()
            .map(|r| SearchResultJson {
                id: r.id,
                score: r.score,
                text: r.text,
                metadata: r.metadata,
            })
            .collect(),
        query: req.query,
        took_ms,
    };

    Ok(axum::response::Json(response))
}

#[cfg(not(feature = "server"))]
pub async fn run(_args: ServeArgs, _verbose: bool) -> anyhow::Result<()> {
    anyhow::bail!("Server feature not enabled. Rebuild with --features server")
}
