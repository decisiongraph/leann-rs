//! LLM module - language model providers for RAG

mod ollama;
mod openai;
mod anthropic;

use tracing::info;

/// LLM provider type
#[derive(Debug, Clone)]
pub enum LlmType {
    Ollama { host: Option<String> },
    OpenAI { api_key: Option<String>, base_url: Option<String> },
    Anthropic { api_key: Option<String>, base_url: Option<String> },
}

/// Unified LLM provider
pub struct LlmProvider {
    model_name: String,
    inner: LlmProviderInner,
}

enum LlmProviderInner {
    Ollama(ollama::OllamaLlm),
    OpenAI(openai::OpenAILlm),
    Anthropic(anthropic::AnthropicLlm),
}

impl LlmProvider {
    /// Create a new LLM provider
    pub fn new(model_name: String, llm_type: LlmType) -> anyhow::Result<Self> {
        let inner = match llm_type {
            LlmType::Ollama { host } => {
                LlmProviderInner::Ollama(ollama::OllamaLlm::new(model_name.clone(), host)?)
            }
            LlmType::OpenAI { api_key, base_url } => {
                LlmProviderInner::OpenAI(openai::OpenAILlm::new(model_name.clone(), api_key, base_url)?)
            }
            LlmType::Anthropic { api_key, base_url } => {
                LlmProviderInner::Anthropic(anthropic::AnthropicLlm::new(model_name.clone(), api_key, base_url)?)
            }
        };

        info!("Initialized LLM provider: {}", model_name);

        Ok(Self { model_name, inner })
    }

    /// Generate a response
    pub async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        match &self.inner {
            LlmProviderInner::Ollama(llm) => llm.generate(prompt).await,
            LlmProviderInner::OpenAI(llm) => llm.generate(prompt).await,
            LlmProviderInner::Anthropic(llm) => llm.generate(prompt).await,
        }
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}
