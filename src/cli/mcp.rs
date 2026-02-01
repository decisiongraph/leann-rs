//! MCP server - Model Context Protocol integration for Claude Code
//!
//! Provides LEANN search capabilities as MCP tools.

use std::path::PathBuf;

use clap::Args;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use tokio::io::{stdin, stdout};
use tracing::info;

use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{find_index, IndexMeta, IndexSearcher, MetadataFilter, RecomputeSearcher, SearchOptions};

#[derive(Args)]
pub struct McpArgs {
    /// Index name to expose via MCP (optional, can also specify per-request)
    #[arg(long)]
    pub index: Option<String>,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,
}

/// Input parameters for search tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchInput {
    /// Search query - the text to search for semantically
    #[schemars(description = "The search query text")]
    query: String,

    /// Index name to search (optional if default index is set)
    #[serde(default)]
    #[schemars(description = "Name of the LEANN index to search")]
    index: Option<String>,

    /// Number of results to return (default: 5)
    #[serde(default = "default_top_k")]
    #[schemars(description = "Number of results to return")]
    top_k: usize,

    /// Filter expression (e.g., "source:*.rs" or "type=code")
    #[serde(default)]
    #[schemars(description = "Metadata filter expression")]
    filter: Option<String>,

    /// Enable hybrid search combining vector + keyword matching
    #[serde(default)]
    #[schemars(description = "Enable hybrid vector + BM25 search")]
    hybrid: bool,
}

fn default_top_k() -> usize {
    5
}

/// Input parameters for list_indexes tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListIndexesInput {}

/// LEANN MCP Server
#[derive(Clone, Debug)]
pub struct LeannMcpServer {
    default_index: Option<String>,
    embedding_api_key: Option<String>,
    embedding_api_base: Option<String>,
    embedding_host: Option<String>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl LeannMcpServer {
    fn new(
        default_index: Option<String>,
        embedding_api_key: Option<String>,
        embedding_api_base: Option<String>,
        embedding_host: Option<String>,
    ) -> Self {
        Self {
            default_index,
            embedding_api_key,
            embedding_api_base,
            embedding_host,
            tool_router: Self::tool_router(),
        }
    }

    /// Search for documents using semantic similarity
    #[tool(description = "Search for documents in the LEANN vector database using semantic similarity. Returns relevant passages with scores and source metadata.")]
    async fn search(
        &self,
        params: Parameters<SearchInput>,
    ) -> Result<CallToolResult, McpError> {
        match self.do_search(params.0).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Search error: {}",
                e
            ))])),
        }
    }

    /// List all available LEANN indexes
    #[tool(description = "List all available LEANN indexes (both local project and global user indexes).")]
    async fn list_indexes(
        &self,
        _params: Parameters<ListIndexesInput>,
    ) -> Result<CallToolResult, McpError> {
        match self.do_list_indexes() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Error listing indexes: {}",
                e
            ))])),
        }
    }
}

impl LeannMcpServer {
    async fn do_search(&self, input: SearchInput) -> anyhow::Result<String> {
        let index_name = input
            .index
            .as_ref()
            .or(self.default_index.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No index specified. Use --index flag or provide 'index' in the request."
                )
            })?;

        // Find index
        let index_dir = find_index(index_name)?;
        let meta_path = index_dir.join("documents.leann.meta.json");
        let index_path = index_dir.join("documents.leann");

        // Load metadata
        let meta = IndexMeta::load(&meta_path)?;

        // Create embedding provider
        let embedding_mode = match meta.embedding_mode.as_str() {
            "openai" => EmbeddingMode::OpenAI {
                api_key: self.embedding_api_key.clone(),
                base_url: self.embedding_api_base.clone(),
            },
            "ollama" => EmbeddingMode::Ollama {
                host: self.embedding_host.clone(),
            },
            "gemini" => EmbeddingMode::Gemini {
                api_key: std::env::var("GOOGLE_API_KEY").ok(),
            },
            _ => anyhow::bail!("Unknown embedding mode: {}", meta.embedding_mode),
        };

        let embedding_provider =
            EmbeddingProvider::new(meta.embedding_model.clone(), embedding_mode).await?;

        // Compute query embedding
        let query_embedding = embedding_provider.embed(&[&input.query]).await?;
        let query_embedding = &query_embedding[0];

        // Parse filter
        let filter = input.filter.as_ref().and_then(|s| MetadataFilter::parse(s));

        // Search
        let results = if meta.is_pruned {
            let searcher = RecomputeSearcher::load(&index_path, meta.dimensions)?;
            searcher
                .search(query_embedding, &embedding_provider, input.top_k, filter.as_ref())
                .await?
        } else {
            let searcher = IndexSearcher::load(&index_path, &meta)?;
            let mut opts = SearchOptions::new(input.top_k, 64);

            if let Some(f) = filter {
                opts = opts.with_filter(f);
            }

            if input.hybrid {
                opts = opts.with_hybrid(input.query.clone(), 0.7);
            }

            searcher.search_with_options(query_embedding, &opts)?
        };

        // Format results as markdown
        let mut output = format!("## Search Results for \"{}\"\n\n", input.query);
        output.push_str(&format!(
            "Found {} results in index '{}'.\n\n",
            results.len(),
            index_name
        ));

        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "### Result {} (score: {:.4})\n\n",
                i + 1,
                result.score
            ));

            if let Some(source) = result.metadata.get("source") {
                if let Some(s) = source.as_str() {
                    output.push_str(&format!("**Source:** `{}`\n\n", s));
                }
            }

            output.push_str(&result.text);
            output.push_str("\n\n---\n\n");
        }

        Ok(output)
    }

    fn do_list_indexes(&self) -> anyhow::Result<String> {
        let mut indexes = Vec::new();

        // Check local .leann directory
        let local_path = PathBuf::from(".leann").join("indexes");
        if local_path.exists() {
            if let Ok(entries) = std::fs::read_dir(&local_path) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            indexes.push(format!("{} (local)", name));
                        }
                    }
                }
            }
        }

        // Check global ~/.leann directory
        if let Some(home) = dirs::home_dir() {
            let global_path = home.join(".leann").join("indexes");
            if global_path.exists() {
                if let Ok(entries) = std::fs::read_dir(&global_path) {
                    for entry in entries.flatten() {
                        if entry.path().is_dir() {
                            if let Some(name) = entry.file_name().to_str() {
                                indexes.push(format!("{} (global)", name));
                            }
                        }
                    }
                }
            }
        }

        if indexes.is_empty() {
            Ok("No indexes found. Use `leann build <name> --docs <path>` to create one.".to_string())
        } else {
            let mut output = "## Available Indexes\n\n".to_string();
            for idx in indexes {
                output.push_str(&format!("- {}\n", idx));
            }
            Ok(output)
        }
    }
}

#[tool_handler]
impl ServerHandler for LeannMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "leann-mcp".to_string(),
                title: Some("LEANN MCP Server".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "LEANN vector database MCP server. Use 'search' to find relevant documents \
                 and 'list_indexes' to see available indexes."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}

pub async fn run(args: McpArgs, _verbose: bool) -> anyhow::Result<()> {
    // Initialize logging to stderr (stdout is used for MCP protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting LEANN MCP server...");

    let server = LeannMcpServer::new(
        args.index,
        args.embedding_api_key,
        args.embedding_api_base,
        args.embedding_host,
    );

    // Serve using stdio transport
    let service = server.serve((stdin(), stdout())).await?;

    info!("LEANN MCP server ready");
    service.waiting().await?;

    Ok(())
}
