//! Anthropic LLM provider

use std::env;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::http::{check_response, create_client};

/// Anthropic LLM provider
pub struct AnthropicLlm {
    client: Client,
    api_key: String,
    base_url: String,
    model_name: String,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

impl AnthropicLlm {
    /// Create a new Anthropic LLM provider
    pub fn new(
        model_name: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> anyhow::Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

        let base_url = base_url
            .or_else(|| env::var("ANTHROPIC_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        let client = create_client();

        info!("Anthropic LLM provider: {}", model_name);

        Ok(Self {
            client,
            api_key,
            base_url,
            model_name,
        })
    }

    /// Generate a response
    pub async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let request = AnthropicRequest {
            model: self.model_name.clone(),
            max_tokens: 1000,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let response = check_response(response, "Anthropic").await?;
        let anthropic_response: AnthropicResponse = response.json().await?;

        let content = anthropic_response
            .content
            .iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    block.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(content)
    }
}
