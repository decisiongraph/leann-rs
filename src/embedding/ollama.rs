//! Ollama embedding provider

use std::env;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Ollama embedding provider
pub struct OllamaEmbedding {
    client: Client,
    host: String,
    model_name: String,
    dimensions: usize,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct ModelInfoResponse {
    model_info: Option<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    #[serde(flatten)]
    info: std::collections::HashMap<String, serde_json::Value>,
}

impl OllamaEmbedding {
    /// Create a new Ollama embedding provider
    pub fn new(model_name: String, host: Option<String>) -> anyhow::Result<Self> {
        let host = host
            .or_else(|| env::var("LEANN_OLLAMA_HOST").ok())
            .or_else(|| env::var("OLLAMA_HOST").ok())
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let client = Client::new();

        // Default dimensions for common embedding models
        let dimensions = match model_name.split(':').next().unwrap_or(&model_name) {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            "all-minilm" => 384,
            "bge-m3" => 1024,
            "snowflake-arctic-embed" => 1024,
            _ => 768, // Default
        };

        info!(
            "Ollama embedding provider: {} @ {} ({} dims)",
            model_name, host, dimensions
        );

        Ok(Self {
            client,
            host,
            model_name,
            dimensions,
        })
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

        // Process in batches of 32 (Ollama recommendation)
        let batch_size = 32;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts_vec.chunks(batch_size) {
            let request = EmbedRequest {
                model: self.model_name.clone(),
                input: batch.to_vec(),
            };

            let response = self
                .client
                .post(format!("{}/api/embed", self.host))
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Ollama API error {}: {}", status, body);
            }

            let embed_response: EmbedResponse = response.json().await?;
            all_embeddings.extend(embed_response.embeddings);
        }

        Ok(all_embeddings)
    }
}
