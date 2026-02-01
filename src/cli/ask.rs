//! Ask command - RAG question answering

use std::path::PathBuf;

use clap::Args;
use tracing::info;

use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{IndexMeta, IndexSearcher};
use crate::llm::{LlmProvider, LlmType};

#[derive(Args)]
pub struct AskArgs {
    /// Index name to query
    pub index_name: String,

    /// Question to ask (omit for interactive mode)
    pub query: Option<String>,

    /// LLM provider
    #[arg(long, default_value = "ollama", value_parser = ["ollama", "openai", "anthropic", "simulated"])]
    pub llm: String,

    /// LLM model name
    #[arg(long, default_value = "qwen3:8b")]
    pub model: String,

    /// Ollama host
    #[arg(long, env = "OLLAMA_HOST")]
    pub host: Option<String>,

    /// OpenAI/Anthropic API key
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub api_base: Option<String>,

    /// Interactive chat mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Number of passages to retrieve
    #[arg(long, default_value = "5")]
    pub top_k: usize,

    /// Search complexity
    #[arg(long, default_value = "64")]
    pub complexity: usize,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// OpenAI API base URL for embeddings
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,
}

pub async fn run(args: AskArgs, _verbose: bool) -> anyhow::Result<()> {
    // Find index
    let index_dir = find_index(&args.index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    // Load metadata
    let meta = IndexMeta::load(&meta_path)?;

    info!(
        "Using index '{}' ({} passages)",
        args.index_name, meta.passage_count
    );

    // Create embedding provider
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
        _ => anyhow::bail!("Unknown embedding mode: {}", meta.embedding_mode),
    };

    let embedding_provider = EmbeddingProvider::new(
        meta.embedding_model.clone(),
        embedding_mode,
    ).await?;

    // Load index
    let searcher = IndexSearcher::load(&index_path, &meta)?;

    // Create LLM provider
    let llm_type = match args.llm.as_str() {
        "ollama" => LlmType::Ollama {
            host: args.host.clone(),
        },
        "openai" => LlmType::OpenAI {
            api_key: args.api_key.clone(),
            base_url: args.api_base.clone(),
        },
        "anthropic" => LlmType::Anthropic {
            api_key: args.api_key.clone(),
            base_url: args.api_base.clone(),
        },
        "simulated" => LlmType::Simulated,
        _ => anyhow::bail!("Unknown LLM provider: {}", args.llm),
    };

    let llm = LlmProvider::new(args.model.clone(), llm_type)?;

    println!("Using {} with model {}", args.llm, args.model);

    if args.interactive {
        run_interactive(
            &embedding_provider,
            &searcher,
            &llm,
            args.top_k,
            args.complexity,
        ).await
    } else {
        let query = args.query.ok_or_else(|| {
            anyhow::anyhow!("Query required in non-interactive mode. Use -i for interactive mode.")
        })?;

        let answer = ask_question(
            &query,
            &embedding_provider,
            &searcher,
            &llm,
            args.top_k,
            args.complexity,
        ).await?;

        println!("\nAnswer:\n{}", answer);
        Ok(())
    }
}

async fn ask_question(
    query: &str,
    embedding_provider: &EmbeddingProvider,
    searcher: &IndexSearcher,
    llm: &LlmProvider,
    top_k: usize,
    complexity: usize,
) -> anyhow::Result<String> {
    // Compute query embedding
    let query_embedding = embedding_provider.embed(&[query]).await?;
    let query_embedding = &query_embedding[0];

    // Search for relevant passages
    let results = searcher.search(query_embedding, top_k, complexity)?;

    if results.is_empty() {
        return Ok("No relevant passages found.".to_string());
    }

    // Build context from results
    let context: String = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[{}] {}", i + 1, r.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    // Build prompt
    let prompt = format!(
        r#"Here is some retrieved context that might help answer your question:

{}

Question: {}

Please provide the best answer you can based on this context and your knowledge."#,
        context, query
    );

    // Generate answer
    llm.generate(&prompt).await
}

async fn run_interactive(
    embedding_provider: &EmbeddingProvider,
    searcher: &IndexSearcher,
    llm: &LlmProvider,
    top_k: usize,
    complexity: usize,
) -> anyhow::Result<()> {
    use rustyline::error::ReadlineError;
    use rustyline::{DefaultEditor, Result as RlResult};

    println!("\n🔍 LEANN Interactive Mode");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Commands:");
    println!("  /help     - Show this help message");
    println!("  /clear    - Clear conversation history");
    println!("  /history  - Show command history");
    println!("  /quit     - Exit interactive mode");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Setup readline with history
    let mut rl = DefaultEditor::new()?;

    // Load history from file if it exists
    let history_path = dirs::home_dir()
        .map(|h| h.join(".leann").join("history.txt"))
        .unwrap_or_else(|| std::path::PathBuf::from(".leann_history"));

    if history_path.exists() {
        let _ = rl.load_history(&history_path);
    }

    let mut conversation_history: Vec<(String, String)> = Vec::new();

    loop {
        let readline = rl.readline("You: ");

        match readline {
            Ok(line) => {
                let input = line.trim();

                if input.is_empty() {
                    continue;
                }

                // Add to readline history
                let _ = rl.add_history_entry(input);

                // Handle commands
                if input.starts_with('/') {
                    match input {
                        "/help" | "/h" | "/?" => {
                            println!("\nCommands:");
                            println!("  /help     - Show this help message");
                            println!("  /clear    - Clear conversation history");
                            println!("  /history  - Show command history");
                            println!("  /quit     - Exit interactive mode\n");
                            continue;
                        }
                        "/clear" | "/c" => {
                            conversation_history.clear();
                            println!("\nConversation history cleared.\n");
                            continue;
                        }
                        "/history" | "/hist" => {
                            println!("\nConversation history:");
                            if conversation_history.is_empty() {
                                println!("  (empty)");
                            } else {
                                for (i, (q, _)) in conversation_history.iter().enumerate() {
                                    println!("  {}. {}", i + 1, q);
                                }
                            }
                            println!();
                            continue;
                        }
                        "/quit" | "/q" | "/exit" => {
                            println!("\nGoodbye!");
                            break;
                        }
                        _ => {
                            println!("\nUnknown command: {}. Type /help for available commands.\n", input);
                            continue;
                        }
                    }
                }

                // Regular question
                match ask_question(input, embedding_provider, searcher, llm, top_k, complexity).await {
                    Ok(answer) => {
                        println!("\nLEANN: {}\n", answer);
                        conversation_history.push((input.to_string(), answer));
                    }
                    Err(e) => eprintln!("\nError: {}\n", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nInterrupted. Type /quit to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    if let Some(parent) = history_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = rl.save_history(&history_path);

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
