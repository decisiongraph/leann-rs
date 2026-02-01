//! Token truncation using tiktoken

use tiktoken_rs::cl100k_base;
use tracing::warn;

/// Token limits for common embedding models
pub fn get_token_limit(model_name: &str) -> usize {
    let base_name = model_name.split(':').next().unwrap_or(model_name);

    match base_name {
        // OpenAI models
        "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002" => 8192,

        // Ollama/local models
        "nomic-embed-text" | "nomic-embed-text-v1.5" => 2048,
        "nomic-embed-text-v2" => 512,
        "mxbai-embed-large" => 512,
        "all-minilm" => 512,
        "bge-m3" => 8192,
        "snowflake-arctic-embed" => 512,

        // Default fallback
        _ => 2048,
    }
}

/// Truncate texts to fit within token limit
pub fn truncate_to_token_limit(texts: &[String], token_limit: usize) -> Vec<String> {
    if texts.is_empty() {
        return Vec::new();
    }

    let bpe = cl100k_base().expect("Failed to load tiktoken encoding");

    let mut truncated = Vec::with_capacity(texts.len());
    let mut truncation_count = 0;
    let mut total_tokens_removed = 0;

    for (i, text) in texts.iter().enumerate() {
        let tokens = bpe.encode_with_special_tokens(text);
        let original_length = tokens.len();

        if original_length <= token_limit {
            truncated.push(text.clone());
        } else {
            // Truncate to limit
            let truncated_tokens: Vec<u32> = tokens.into_iter().take(token_limit).collect();
            let truncated_text = bpe
                .decode(truncated_tokens)
                .unwrap_or_else(|_| text[..text.len().min(token_limit * 4)].to_string());

            truncated.push(truncated_text);

            truncation_count += 1;
            total_tokens_removed += original_length - token_limit;

            if truncation_count <= 3 {
                warn!(
                    "Text {} truncated: {} → {} tokens ({} removed)",
                    i + 1,
                    original_length,
                    token_limit,
                    original_length - token_limit
                );
            }
        }
    }

    if truncation_count > 3 {
        warn!(
            "Truncation summary: {}/{} texts truncated ({} tokens removed total)",
            truncation_count,
            texts.len(),
            total_tokens_removed
        );
    }

    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_limit_lookup() {
        assert_eq!(get_token_limit("text-embedding-3-small"), 8192);
        assert_eq!(get_token_limit("nomic-embed-text"), 2048);
        assert_eq!(get_token_limit("nomic-embed-text:latest"), 2048);
        assert_eq!(get_token_limit("unknown-model"), 2048);
    }

    #[test]
    fn test_truncation() {
        let short_text = "Hello world".to_string();
        let result = truncate_to_token_limit(&[short_text.clone()], 100);
        assert_eq!(result[0], short_text);
    }
}
