//! OpenAI LLM provider

use std::env;

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use tracing::info;

/// OpenAI LLM provider
pub struct OpenAILlm {
    client: Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAILlm {
    /// Create a new OpenAI LLM provider
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

        info!("OpenAI LLM provider: {}", model_name);

        Ok(Self { client, model_name })
    }

    /// Generate a response
    pub async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model_name)
            .messages([ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into()])
            .max_tokens(1000u32)
            .build()?;

        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_default();

        Ok(content)
    }
}
