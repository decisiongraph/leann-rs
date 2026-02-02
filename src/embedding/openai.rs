//! OpenAI embedding provider

use std::env;

use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use tracing::info;

/// OpenAI embedding provider
pub struct OpenAIEmbedding {
    client: Client<OpenAIConfig>,
    model_name: String,
    dimensions: usize,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding provider
    pub fn new(
        model_name: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> anyhow::Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

        let mut config = OpenAIConfig::new().with_api_key(api_key);

        if let Some(base_url) = base_url.or_else(|| env::var("OPENAI_BASE_URL").ok()) {
            config = config.with_api_base(base_url);
        }

        let client = Client::with_config(config);

        // Determine dimensions based on model name (will be verified/updated on first embed call)
        let dimensions = match model_name.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 0, // Unknown - will auto-detect on first call
        };

        info!("OpenAI embedding provider: {} ({} dims)", model_name,
              if dimensions == 0 { "auto".to_string() } else { dimensions.to_string() });

        Ok(Self {
            client,
            model_name,
            dimensions,
        })
    }

    /// Create a new OpenAI embedding provider and detect dimensions
    pub async fn new_with_detection(
        model_name: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> anyhow::Result<Self> {
        let mut provider = Self::new(model_name, api_key, base_url)?;

        // If dimensions unknown, detect by doing a test embedding
        if provider.dimensions == 0 {
            let test = provider.embed(&["test"]).await?;
            if let Some(embedding) = test.first() {
                provider.dimensions = embedding.len();
                info!("Auto-detected embedding dimensions: {}", provider.dimensions);
            }
        }

        Ok(provider)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Compute embeddings
    pub async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let texts_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Process in batches of 100 (OpenAI limit)
        let batch_size = 100;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts_vec.chunks(batch_size) {
            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model_name)
                .input(EmbeddingInput::StringArray(batch.to_vec()))
                .build()?;

            let response = self.client.embeddings().create(request).await?;

            for embedding_data in response.data {
                all_embeddings.push(embedding_data.embedding);
            }
        }

        Ok(all_embeddings)
    }
}
