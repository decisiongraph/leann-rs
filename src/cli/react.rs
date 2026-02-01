//! ReAct agent command - multi-turn reasoning with tools

use clap::Args;
use tracing::info;

use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::index::{find_index, IndexMeta, IndexSearcher};
use crate::llm::{LlmProvider, LlmType};

#[derive(Args)]
pub struct ReactArgs {
    /// Index name to query
    pub index_name: String,

    /// Question to answer
    pub query: String,

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

    /// Maximum reasoning steps
    #[arg(long, default_value = "5")]
    pub max_steps: usize,

    /// Number of passages to retrieve per search
    #[arg(long, default_value = "3")]
    pub top_k: usize,

    /// Show reasoning trace
    #[arg(long)]
    pub verbose: bool,

    /// API key for embedding service
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,
}

/// ReAct agent state
struct ReActAgent<'a> {
    embedding_provider: &'a EmbeddingProvider,
    searcher: &'a IndexSearcher,
    llm: &'a LlmProvider,
    top_k: usize,
    max_steps: usize,
    verbose: bool,
}

impl<'a> ReActAgent<'a> {
    fn new(
        embedding_provider: &'a EmbeddingProvider,
        searcher: &'a IndexSearcher,
        llm: &'a LlmProvider,
        top_k: usize,
        max_steps: usize,
        verbose: bool,
    ) -> Self {
        Self {
            embedding_provider,
            searcher,
            llm,
            top_k,
            max_steps,
            verbose,
        }
    }

    async fn run(&self, query: &str) -> anyhow::Result<String> {
        let system_prompt = r#"You are a helpful assistant that answers questions using available tools.

Available tools:
1. search(query) - Search the knowledge base for relevant information
2. finish(answer) - Provide the final answer

For each step, use the following format:
Thought: [Your reasoning about what to do next]
Action: [tool_name(argument)]

After getting search results, you'll see:
Observation: [results from the tool]

Continue until you have enough information, then use finish(answer) to provide your final answer.

Important:
- Always search for relevant information before answering
- If the first search doesn't give enough info, try different search queries
- Be concise in your final answer
"#;

        let mut history = format!(
            "{}\n\nQuestion: {}\n\nLet me search for relevant information.\n",
            system_prompt, query
        );

        for step in 0..self.max_steps {
            if self.verbose {
                println!("\n--- Step {} ---", step + 1);
            }

            // Get next action from LLM
            let response = self.llm.generate(&history).await?;

            if self.verbose {
                println!("LLM: {}", response);
            }

            history.push_str(&response);
            history.push('\n');

            // Parse action
            if let Some(action) = self.parse_action(&response) {
                match action {
                    Action::Search(search_query) => {
                        if self.verbose {
                            println!("Searching: {}", search_query);
                        }

                        let observation = self.execute_search(&search_query).await?;
                        history.push_str(&format!("Observation: {}\n\n", observation));

                        if self.verbose {
                            println!("Observation: {}", observation);
                        }
                    }
                    Action::Finish(answer) => {
                        return Ok(answer);
                    }
                }
            } else {
                // No valid action found, try to extract answer
                if response.to_lowercase().contains("final answer")
                    || response.to_lowercase().contains("the answer is")
                {
                    return Ok(response);
                }

                // Prompt for action
                history.push_str("Please use an action: search(query) or finish(answer)\n");
            }
        }

        // Max steps reached, ask for final answer
        history.push_str("Maximum steps reached. Please provide your final answer using finish(answer).\n");
        let final_response = self.llm.generate(&history).await?;

        if let Some(Action::Finish(answer)) = self.parse_action(&final_response) {
            Ok(answer)
        } else {
            Ok(final_response)
        }
    }

    fn parse_action(&self, response: &str) -> Option<Action> {
        // Look for search(query) pattern
        if let Some(start) = response.find("search(") {
            let rest = &response[start + 7..];
            if let Some(end) = rest.find(')') {
                let query = rest[..end].trim().trim_matches('"').trim_matches('\'');
                return Some(Action::Search(query.to_string()));
            }
        }

        // Look for finish(answer) pattern
        if let Some(start) = response.find("finish(") {
            let rest = &response[start + 7..];
            if let Some(end) = rest.rfind(')') {
                let answer = rest[..end].trim().trim_matches('"').trim_matches('\'');
                return Some(Action::Finish(answer.to_string()));
            }
        }

        // Look for Action: search pattern
        for line in response.lines() {
            let line = line.trim();
            if line.starts_with("Action:") {
                let action_str = line["Action:".len()..].trim();
                if action_str.starts_with("search") {
                    if let Some(start) = action_str.find('(') {
                        let rest = &action_str[start + 1..];
                        if let Some(end) = rest.find(')') {
                            let query = rest[..end].trim().trim_matches('"').trim_matches('\'');
                            return Some(Action::Search(query.to_string()));
                        }
                    }
                } else if action_str.starts_with("finish") {
                    if let Some(start) = action_str.find('(') {
                        let rest = &action_str[start + 1..];
                        if let Some(end) = rest.rfind(')') {
                            let answer = rest[..end].trim().trim_matches('"').trim_matches('\'');
                            return Some(Action::Finish(answer.to_string()));
                        }
                    }
                }
            }
        }

        None
    }

    async fn execute_search(&self, query: &str) -> anyhow::Result<String> {
        let query_embedding = self.embedding_provider.embed(&[query]).await?;
        let results = self.searcher.search(&query_embedding[0], self.top_k, 64)?;

        if results.is_empty() {
            return Ok("No relevant results found.".to_string());
        }

        let mut observation = String::new();
        for (i, result) in results.iter().enumerate() {
            let snippet = if result.text.len() > 300 {
                format!("{}...", &result.text[..300])
            } else {
                result.text.clone()
            };
            observation.push_str(&format!("[{}] {}\n", i + 1, snippet));
        }

        Ok(observation)
    }
}

enum Action {
    Search(String),
    Finish(String),
}

pub async fn run(args: ReactArgs, _verbose: bool) -> anyhow::Result<()> {
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

    println!("ReAct Agent using {} with model {}", args.llm, args.model);
    println!("Question: {}\n", args.query);

    // Run agent
    let agent = ReActAgent::new(
        &embedding_provider,
        &searcher,
        &llm,
        args.top_k,
        args.max_steps,
        args.verbose,
    );

    let answer = agent.run(&args.query).await?;

    println!("\n=== Final Answer ===\n{}", answer);

    Ok(())
}

