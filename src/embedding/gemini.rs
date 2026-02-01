//! Google Gemini embedding provider

use std::env;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::http::create_client;

/// Gemini embedding provider
pub struct GeminiEmbedding {
    client: Client,
    model_name: String,
    api_key: String,
    dimensions: usize,
}

#[derive(Serialize)]
struct EmbedContentRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct EmbedContentResponse {
    embedding: Embedding,
}

#[derive(Deserialize)]
struct Embedding {
    values: Vec<f32>,
}

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<EmbedRequest>,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<Embedding>,
}

impl GeminiEmbedding {
    /// Create a new Gemini embedding provider
    pub fn new(model_name: String, api_key: Option<String>) -> anyhow::Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("GOOGLE_API_KEY").ok())
            .or_else(|| env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| anyhow::anyhow!("GOOGLE_API_KEY or GEMINI_API_KEY not set"))?;

        let client = create_client();

        // Gemini embedding dimensions
        let dimensions = match model_name.as_str() {
            "text-embedding-004" => 768,
            "embedding-001" => 768,
            _ => 768, // Default for Gemini models
        };

        info!(
            "Gemini embedding provider: {} ({} dims)",
            model_name, dimensions
        );

        Ok(Self {
            client,
            model_name,
            api_key,
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

        // Use batch API for multiple texts
        if texts.len() > 1 {
            self.batch_embed(texts).await
        } else {
            let embedding = self.single_embed(texts[0]).await?;
            Ok(vec![embedding])
        }
    }

    async fn single_embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}",
            self.model_name, self.api_key
        );

        let request = EmbedContentRequest {
            model: format!("models/{}", self.model_name),
            content: Content {
                parts: vec![Part {
                    text: text.to_string(),
                }],
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<EmbedContentResponse>()
            .await?;

        Ok(response.embedding.values)
    }

    async fn batch_embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model_name, self.api_key
        );

        // Gemini batch limit is 100
        let batch_size = 100;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let requests: Vec<EmbedRequest> = batch
                .iter()
                .map(|text| EmbedRequest {
                    model: format!("models/{}", self.model_name),
                    content: Content {
                        parts: vec![Part {
                            text: text.to_string(),
                        }],
                    },
                })
                .collect();

            let request = BatchEmbedRequest { requests };

            let response = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await?
                .error_for_status()?
                .json::<BatchEmbedResponse>()
                .await?;

            for embedding in response.embeddings {
                all_embeddings.push(embedding.values);
            }
        }

        Ok(all_embeddings)
    }
}
