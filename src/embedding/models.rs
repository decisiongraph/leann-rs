//! Embedding model registry with model-specific configurations
//!
//! Different embedding models have different requirements:
//! - Some need prefixes for documents vs queries (asymmetric)
//! - Some produce normalized embeddings (use cosine)
//! - Some have specific token limits

/// Model configuration for embedding
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Prefix to add to documents during indexing
    pub document_prefix: &'static str,
    /// Prefix to add to queries during search
    pub query_prefix: &'static str,
    /// Whether embeddings are L2 normalized
    pub normalized: bool,
    /// Embedding dimensions
    pub dimensions: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            document_prefix: "",
            query_prefix: "",
            normalized: false,
            dimensions: 768,
        }
    }
}

/// Get model configuration for known models
pub fn get_model_config(model_name: &str) -> ModelConfig {
    // Normalize model name (remove version tags like :latest)
    let base_name = model_name.split(':').next().unwrap_or(model_name);

    match base_name {
        // Nomic models - require search_document/search_query prefixes
        "nomic-embed-text" | "nomic-embed-text-v1" | "nomic-embed-text-v1.5"
        | "text-embedding-nomic-embed-text-v1.5" => ModelConfig {
            document_prefix: "search_document: ",
            query_prefix: "search_query: ",
            normalized: true,
            dimensions: 768,
        },

        // MixedBread mxbai - uses Represent prefixes
        "mxbai-embed-large" | "mxbai-embed-large-v1" => ModelConfig {
            document_prefix: "Represent this document for retrieval: ",
            query_prefix: "Represent this sentence for searching relevant passages: ",
            normalized: true,
            dimensions: 1024,
        },

        // BGE models - use instruction prefixes for queries only
        "bge-small-en" | "bge-base-en" | "bge-large-en"
        | "bge-small-en-v1.5" | "bge-base-en-v1.5" | "bge-large-en-v1.5" => ModelConfig {
            document_prefix: "",
            query_prefix: "Represent this sentence for searching relevant passages: ",
            normalized: true,
            dimensions: match base_name {
                s if s.contains("small") => 384,
                s if s.contains("large") => 1024,
                _ => 768,
            },
        },

        // E5 models - use query/passage prefixes
        "e5-small" | "e5-base" | "e5-large"
        | "e5-small-v2" | "e5-base-v2" | "e5-large-v2"
        | "multilingual-e5-small" | "multilingual-e5-base" | "multilingual-e5-large" => ModelConfig {
            document_prefix: "passage: ",
            query_prefix: "query: ",
            normalized: true,
            dimensions: match base_name {
                s if s.contains("small") => 384,
                s if s.contains("large") => 1024,
                _ => 768,
            },
        },

        // GTE models - no prefix needed
        "gte-small" | "gte-base" | "gte-large" => ModelConfig {
            document_prefix: "",
            query_prefix: "",
            normalized: true,
            dimensions: match base_name {
                s if s.contains("small") => 384,
                s if s.contains("large") => 1024,
                _ => 768,
            },
        },

        // All-MiniLM - no prefix needed
        "all-minilm" | "all-MiniLM-L6-v2" | "all-MiniLM-L12-v2" => ModelConfig {
            document_prefix: "",
            query_prefix: "",
            normalized: true,
            dimensions: 384,
        },

        // OpenAI models - no prefix needed
        "text-embedding-3-small" => ModelConfig {
            document_prefix: "",
            query_prefix: "",
            normalized: true,
            dimensions: 1536,
        },
        "text-embedding-3-large" => ModelConfig {
            document_prefix: "",
            query_prefix: "",
            normalized: true,
            dimensions: 3072,
        },
        "text-embedding-ada-002" => ModelConfig {
            document_prefix: "",
            query_prefix: "",
            normalized: true,
            dimensions: 1536,
        },

        // Default for unknown models
        _ => ModelConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nomic_config() {
        let config = get_model_config("nomic-embed-text");
        assert_eq!(config.document_prefix, "search_document: ");
        assert_eq!(config.query_prefix, "search_query: ");
        assert!(config.normalized);
    }

    #[test]
    fn test_nomic_with_version() {
        let config = get_model_config("nomic-embed-text:latest");
        assert_eq!(config.document_prefix, "search_document: ");
    }

    #[test]
    fn test_mxbai_config() {
        let config = get_model_config("mxbai-embed-large");
        assert_eq!(config.dimensions, 1024);
        assert!(!config.query_prefix.is_empty());
    }

    #[test]
    fn test_unknown_model() {
        let config = get_model_config("some-unknown-model");
        assert_eq!(config.document_prefix, "");
        assert_eq!(config.query_prefix, "");
    }
}
