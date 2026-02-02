//! CLI module - command definitions and handlers

pub mod build;
mod search;
mod ask;
mod list;
mod remove;
mod react;
mod serve;
mod update;
mod prune;
mod config_cmd;
#[cfg(feature = "mcp")]
mod mcp;

use clap::{Parser, Subcommand};

pub use build::BuildArgs;
pub use search::SearchArgs;
pub use ask::AskArgs;
pub use list::ListArgs;
pub use remove::RemoveArgs;
pub use react::ReactArgs;
pub use serve::ServeArgs;
pub use update::UpdateArgs;
pub use prune::PruneArgs;
pub use config_cmd::ConfigArgs;
#[cfg(feature = "mcp")]
pub use mcp::McpArgs;

/// LEANN - Lightweight vector database for RAG
#[derive(Parser)]
#[command(name = "leann")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress non-essential output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Build a document index
    Build(BuildArgs),

    /// Update an existing index with new documents
    Update(UpdateArgs),

    /// Search documents in an index
    Search(SearchArgs),

    /// Ask questions using RAG
    Ask(AskArgs),

    /// ReAct agent for multi-turn reasoning
    React(ReactArgs),

    /// Start HTTP API server
    Serve(ServeArgs),

    /// List all indexes
    List(ListArgs),

    /// Remove an index
    Remove(RemoveArgs),

    /// Prune embeddings to enable recomputation mode
    Prune(PruneArgs),

    /// Manage configuration
    Config(ConfigArgs),

    /// Start MCP server for Claude Code integration
    #[cfg(feature = "mcp")]
    Mcp(McpArgs),
}

impl Cli {
    pub async fn run(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Build(args) => build::run(args, self.verbose).await,
            Commands::Update(args) => update::run(args, self.verbose).await,
            Commands::Search(args) => search::run(args, self.verbose).await,
            Commands::Ask(args) => ask::run(args, self.verbose).await,
            Commands::React(args) => react::run(args, self.verbose).await,
            Commands::Serve(args) => serve::run(args, self.verbose).await,
            Commands::List(args) => list::run(args).await,
            Commands::Remove(args) => remove::run(args).await,
            Commands::Prune(args) => prune::run(args).await,
            Commands::Config(args) => config_cmd::run(args).await,
            #[cfg(feature = "mcp")]
            Commands::Mcp(args) => mcp::run(args, self.verbose).await,
        }
    }
}
