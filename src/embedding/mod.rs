//! Embedding module - compute embeddings from text

mod openai;
mod ollama;
mod gemini;
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
    Gemini {
        api_key: Option<String>,
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
    Gemini(gemini::GeminiEmbedding),
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
            EmbeddingMode::Gemini { api_key } => {
                let provider = gemini::GeminiEmbedding::new(
                    model_name.clone(),
                    api_key,
                )?;
                let dims = provider.dimensions();
                (EmbeddingProviderInner::Gemini(provider), dims)
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
            EmbeddingProviderInner::Gemini(p) => p.embed(texts).await,
            #[cfg(feature = "local-embeddings")]
            EmbeddingProviderInner::Local(p) => p.embed(texts),
        }
    }

    /// Compute embeddings with a prompt template prefix
    ///
    /// Useful for asymmetric embedding models like E5, BGE, or Instructor
    /// that expect prefixes like "query: " or "passage: "
    pub async fn embed_with_template(
        &self,
        texts: &[&str],
        template: &str,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if template.is_empty() {
            return self.embed(texts).await;
        }

        // Apply template prefix to all texts
        let templated: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", template, t))
            .collect();

        let refs: Vec<&str> = templated.iter().map(|s| s.as_str()).collect();
        self.embed(&refs).await
    }
}

/// Common prompt templates for asymmetric embedding models
pub mod templates {
    /// E5 model query prefix
    pub const E5_QUERY: &str = "query: ";
    /// E5 model passage prefix
    pub const E5_PASSAGE: &str = "passage: ";

    /// BGE model query prefix
    pub const BGE_QUERY: &str = "Represent this sentence for searching relevant passages: ";

    /// Instructor model query prefix
    pub const INSTRUCTOR_QUERY: &str = "Represent the question for retrieving evidence: ";
    /// Instructor model passage prefix
    pub const INSTRUCTOR_PASSAGE: &str = "Represent the document for retrieval: ";
}
