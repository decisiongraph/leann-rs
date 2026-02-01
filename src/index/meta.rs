//! Index metadata handling

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Index metadata stored alongside the index files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    /// Metadata format version
    pub version: String,

    /// Backend used (hnsw, diskann)
    pub backend_name: String,

    /// Embedding model name
    pub embedding_model: String,

    /// Embedding mode (openai, ollama, etc.)
    pub embedding_mode: String,

    /// Embedding dimensions
    pub dimensions: usize,

    /// Total number of passages
    pub passage_count: usize,

    /// Optional backend-specific configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_kwargs: Option<serde_json::Value>,

    /// Optional embedding provider options
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_options: Option<serde_json::Value>,

    /// Whether this index supports embedding recomputation
    #[serde(default)]
    pub is_recompute: bool,

    /// Whether embeddings have been pruned (deleted to save space)
    #[serde(default)]
    pub is_pruned: bool,
}

impl IndexMeta {
    /// Load metadata from a JSON file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let meta: IndexMeta = serde_json::from_str(&content)?;
        Ok(meta)
    }

    /// Save metadata to a JSON file
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
