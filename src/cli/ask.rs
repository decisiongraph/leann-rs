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
    #[arg(long, default_value = "ollama", value_parser = ["ollama", "openai", "anthropic"])]
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
    use std::io::{self, BufRead, Write};

    println!("\nInteractive mode. Type 'quit' or 'exit' to leave.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("You: ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }

        match ask_question(input, embedding_provider, searcher, llm, top_k, complexity).await {
            Ok(answer) => println!("\nLEANN: {}\n", answer),
            Err(e) => eprintln!("\nError: {}\n", e),
        }
    }

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
