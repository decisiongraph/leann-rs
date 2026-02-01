//! Ollama LLM provider

use std::env;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::http::{check_response, create_client};

/// Ollama LLM provider
pub struct OllamaLlm {
    client: Client,
    host: String,
    model_name: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    done: bool,
}

impl OllamaLlm {
    /// Create a new Ollama LLM provider
    pub fn new(model_name: String, host: Option<String>) -> anyhow::Result<Self> {
        let host = host
            .or_else(|| env::var("LEANN_OLLAMA_HOST").ok())
            .or_else(|| env::var("OLLAMA_HOST").ok())
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let client = create_client();

        info!("Ollama LLM provider: {} @ {}", model_name, host);

        Ok(Self {
            client,
            host,
            model_name,
        })
    }

    /// Generate a response
    pub async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let request = GenerateRequest {
            model: self.model_name.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/api/generate", self.host))
            .json(&request)
            .send()
            .await?;

        let response = check_response(response, "Ollama").await?;
        let text = response.text().await?;

        // Parse streaming-style response (multiple JSON objects)
        let mut full_response = String::new();
        for line in text.lines() {
            if !line.is_empty() {
                if let Ok(resp) = serde_json::from_str::<GenerateResponse>(line) {
                    full_response.push_str(&resp.response);
                    if resp.done {
                        break;
                    }
                }
            }
        }

        Ok(full_response)
    }
}
