//! Embedding module - compute embeddings from text

mod openai;
mod ollama;
mod traits;
mod truncate;

#[cfg(feature = "local-embeddings")]
mod candle;

pub use traits::EmbeddingProviderTrait;

use tracing::info;

/// Embedding mode configuration
#[derive(Debug, Clone)]
pub enum EmbeddingMode {
    OpenAI {
        api_key: Option<String>,
        base_url: Option<String>,
    },
    Ollama {
        host: Option<String>,
    },
    #[cfg(feature = "local-embeddings")]
    Local {
        model_path: Option<String>,
    },
}

/// Unified embedding provider
pub struct EmbeddingProvider {
    model_name: String,
    dimensions: usize,
    inner: EmbeddingProviderInner,
}

enum EmbeddingProviderInner {
    OpenAI(openai::OpenAIEmbedding),
    Ollama(ollama::OllamaEmbedding),
    #[cfg(feature = "local-embeddings")]
    Local(candle::CandleEmbedding),
}

impl EmbeddingProvider {
    /// Create a new embedding provider
    pub async fn new(model_name: String, mode: EmbeddingMode) -> anyhow::Result<Self> {
        let (inner, dimensions) = match mode {
            EmbeddingMode::OpenAI { api_key, base_url } => {
                let provider = openai::OpenAIEmbedding::new(
                    model_name.clone(),
                    api_key,
                    base_url,
                )?;
                let dims = provider.dimensions();
                (EmbeddingProviderInner::OpenAI(provider), dims)
            }
            EmbeddingMode::Ollama { host } => {
                let provider = ollama::OllamaEmbedding::new(
                    model_name.clone(),
                    host,
                )?;
                let dims = provider.dimensions();
                (EmbeddingProviderInner::Ollama(provider), dims)
            }
            #[cfg(feature = "local-embeddings")]
            EmbeddingMode::Local { model_path } => {
                let provider = candle::CandleEmbedding::new(
                    model_name.clone(),
                    model_path,
                )?;
                let dims = provider.dimensions();
                (EmbeddingProviderInner::Local(provider), dims)
            }
        };

        info!(
            "Initialized embedding provider: {} ({} dims)",
            model_name, dimensions
        );

        Ok(Self {
            model_name,
            dimensions,
            inner,
        })
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Compute embeddings for texts
    pub async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        match &self.inner {
            EmbeddingProviderInner::OpenAI(p) => p.embed(texts).await,
            EmbeddingProviderInner::Ollama(p) => p.embed(texts).await,
            #[cfg(feature = "local-embeddings")]
            EmbeddingProviderInner::Local(p) => p.embed(texts),
        }
    }
}
