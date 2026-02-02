//! Config command - manage LEANN configuration

use clap::{Args, Subcommand};

use crate::config::Config;

#[derive(Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommands,
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Initialize config file with defaults
    Init {
        /// Overwrite existing config
        #[arg(short, long)]
        force: bool,
    },

    /// Show config file path
    Path,
}

pub async fn run(args: ConfigArgs) -> anyhow::Result<()> {
    match args.command {
        ConfigCommands::Show => {
            let config = Config::load();
            let path = Config::config_path();

            if path.exists() {
                println!("Config file: {}", path.display());
            } else {
                println!("Config file: {} (not found, using defaults)", path.display());
            }
            println!();
            println!("[embedding]");
            println!("provider = \"{}\"", config.embedding.provider);
            println!("model = \"{}\"", config.embedding.model);
            if let Some(host) = &config.embedding.host {
                println!("host = \"{}\"", host);
            }
            if let Some(base_url) = &config.embedding.base_url {
                println!("base_url = \"{}\"", base_url);
            }
            if config.embedding.api_key.is_some() {
                println!("api_key = \"***\"");
            }
            if let Some(batch_size) = config.embedding.batch_size {
                println!("batch_size = {}", batch_size);
            }
            println!();
            println!("[build]");
            println!("chunk_size = {}", config.build.chunk_size);
            println!("chunk_overlap = {}", config.build.chunk_overlap);
            println!("max_file_size_kb = {}", config.build.max_file_size_kb);
        }

        ConfigCommands::Init { force } => {
            let path = Config::config_path();

            if path.exists() && !force {
                anyhow::bail!(
                    "Config file already exists at {}. Use --force to overwrite.",
                    path.display()
                );
            }

            Config::create_example_if_missing()?;
            println!("Created config file at {}", path.display());
            println!();
            println!("Edit the file to customize your default embedding provider and model.");
            println!();
            println!("Common configurations:");
            println!();
            println!("  # Ollama (local, recommended)");
            println!("  provider = \"ollama\"");
            println!("  model = \"nomic-embed-text\"  # or \"mxbai-embed-large\"");
            println!();
            println!("  # LM Studio");
            println!("  provider = \"lmstudio\"");
            println!("  model = \"text-embedding-nomic-embed-text-v1.5\"");
            println!("  base_url = \"http://localhost:1234/v1\"");
            println!();
            println!("  # OpenAI");
            println!("  provider = \"openai\"");
            println!("  model = \"text-embedding-3-small\"");
            println!("  # api_key = \"sk-...\"  # or set OPENAI_API_KEY env var");
        }

        ConfigCommands::Path => {
            println!("{}", Config::config_path().display());
        }
    }

    Ok(())
}
