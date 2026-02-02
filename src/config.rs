//! Configuration file support for LEANN
//!
//! Config file location: ~/.config/leann/config.toml
//!
//! Example config:
//! ```toml
//! [embedding]
//! provider = "ollama"  # ollama, lmstudio, openai, gemini
//! model = "nomic-embed-text"
//! host = "http://localhost:11434"  # for ollama
//! # base_url = "http://localhost:1234/v1"  # for lmstudio/openai-compatible
//! # api_key = "sk-..."  # for openai/gemini
//!
//! [build]
//! chunk_size = 256
//! chunk_overlap = 128
//! max_file_size_kb = 1024
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    #[serde(default)]
    pub build: BuildConfig,
}

/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider type: ollama, lmstudio, openai, gemini
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Model name
    #[serde(default = "default_model")]
    pub model: String,

    /// Host for Ollama (e.g., http://localhost:11434)
    pub host: Option<String>,

    /// Base URL for OpenAI-compatible APIs (e.g., http://localhost:1234/v1)
    pub base_url: Option<String>,

    /// API key for OpenAI/Gemini
    pub api_key: Option<String>,

    /// Prompt template for document embeddings
    pub prompt_template: Option<String>,

    /// Batch size for embedding requests
    pub batch_size: Option<usize>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            host: None,
            base_url: None,
            api_key: None,
            prompt_template: None,
            batch_size: None,
        }
    }
}

fn default_provider() -> String {
    "ollama".to_string()
}

fn default_model() -> String {
    "nomic-embed-text".to_string()
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Chunk size in tokens
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Chunk overlap in tokens
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,

    /// Maximum file size in KB
    #[serde(default = "default_max_file_size_kb")]
    pub max_file_size_kb: usize,

    /// File types to include
    pub file_types: Option<Vec<String>>,

    /// File types to exclude
    pub exclude_types: Option<Vec<String>>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            max_file_size_kb: default_max_file_size_kb(),
            file_types: None,
            exclude_types: None,
        }
    }
}

fn default_chunk_size() -> usize {
    256
}

fn default_chunk_overlap() -> usize {
    128
}

fn default_max_file_size_kb() -> usize {
    1024
}

impl Config {
    /// Get the config file path
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("leann")
            .join("config.toml")
    }

    /// Load config from file, returning defaults if not found
    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    match toml::from_str(&content) {
                        Ok(config) => {
                            tracing::debug!("Loaded config from {:?}", path);
                            return config;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse config file: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read config file: {}", e);
                }
            }
        }
        Self::default()
    }

    /// Save config to file
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Create example config file if it doesn't exist
    pub fn create_example_if_missing() -> anyhow::Result<bool> {
        let path = Self::config_path();
        if path.exists() {
            return Ok(false);
        }

        let example = r#"# LEANN Configuration
# Location: ~/.config/leann/config.toml

[embedding]
# Provider: ollama, lmstudio, openai, gemini
provider = "ollama"

# Model name (provider-specific)
# Ollama: nomic-embed-text, mxbai-embed-large
# LM Studio: text-embedding-nomic-embed-text-v1.5, mxbai-embed-large-v1
# OpenAI: text-embedding-3-small, text-embedding-3-large
model = "nomic-embed-text"

# Ollama host (default: http://localhost:11434)
# host = "http://localhost:11434"

# LM Studio / OpenAI-compatible base URL
# base_url = "http://localhost:1234/v1"

# API key (for OpenAI/Gemini, or set OPENAI_API_KEY/GOOGLE_API_KEY env vars)
# api_key = "sk-..."

# Batch size for embedding requests (default: 32 for ollama, 100 for openai)
# batch_size = 32

[build]
# Chunk size in tokens (default: 256)
chunk_size = 256

# Chunk overlap in tokens (default: 128)
chunk_overlap = 128

# Max file size in KB (default: 1024 = 1MB)
max_file_size_kb = 1024

# File types to include (default: common code and doc files)
# file_types = [".md", ".py", ".js", ".ts", ".rs", ".go"]

# File types to exclude
# exclude_types = [".min.js", ".lock"]
"#;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, example)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.embedding.provider, "ollama");
        assert_eq!(config.embedding.model, "nomic-embed-text");
        assert_eq!(config.build.chunk_size, 256);
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
[embedding]
provider = "lmstudio"
model = "mxbai-embed-large-v1"
base_url = "http://localhost:1234/v1"

[build]
chunk_size = 512
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, "lmstudio");
        assert_eq!(config.embedding.model, "mxbai-embed-large-v1");
        assert_eq!(config.build.chunk_size, 512);
    }
}
