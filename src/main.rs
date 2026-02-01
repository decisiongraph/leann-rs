//! LEANN - Lightweight Embedding-based Approximate Nearest Neighbor
//!
//! A single-binary CLI for building, searching, and querying vector indexes.

mod chunker;
mod cli;
mod index;
mod backend;
mod embedding;
mod llm;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "leann=info,warn".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    // Parse CLI args and run
    let cli = Cli::parse();
    cli.run().await
}
