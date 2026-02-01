//! Index builder - constructs vector indexes

use std::path::Path;

use tracing::info;

use crate::backend::{BackendBuilder, BackendType};

use super::embeddings::EmbeddingsWriter;
use super::passages::{Passage, PassageStore};

/// Builder for creating a LEANN index
pub struct IndexBuilder {
    backend_type: BackendType,
    dimensions: usize,
    graph_degree: usize,
    complexity: usize,
    passages: Vec<Passage>,
    embeddings: Vec<Vec<f32>>,
    ids: Vec<String>,
    /// Whether to enable recomputation mode (saves embeddings separately)
    recompute_mode: bool,
}

impl IndexBuilder {
    /// Create a new index builder
    pub fn new(
        backend_type: BackendType,
        dimensions: usize,
        graph_degree: usize,
        complexity: usize,
    ) -> Self {
        Self {
            backend_type,
            dimensions,
            graph_degree,
            complexity,
            passages: Vec::new(),
            embeddings: Vec::new(),
            ids: Vec::new(),
            recompute_mode: false,
        }
    }

    /// Enable recomputation mode (saves embeddings separately)
    pub fn with_recompute_mode(mut self, enabled: bool) -> Self {
        self.recompute_mode = enabled;
        self
    }

    /// Check if recompute mode is enabled
    pub fn is_recompute_mode(&self) -> bool {
        self.recompute_mode
    }

    /// Add a passage with its embedding
    pub fn add_passage(
        &mut self,
        id: &str,
        text: &str,
        embedding: &[f32],
        metadata: serde_json::Value,
    ) -> anyhow::Result<()> {
        if embedding.len() != self.dimensions {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            );
        }

        self.passages.push(Passage {
            id: id.to_string(),
            text: text.to_string(),
            metadata,
        });
        self.embeddings.push(embedding.to_vec());
        self.ids.push(id.to_string());

        Ok(())
    }

    /// Build and save the index
    pub fn build(self, index_path: &Path) -> anyhow::Result<()> {
        info!(
            "Building index with {} passages, {} dimensions{}",
            self.passages.len(),
            self.dimensions,
            if self.recompute_mode { " (recompute mode)" } else { "" }
        );

        // Write passages to JSONL
        let mut passage_writer = PassageStore::create(index_path)?;
        for passage in &self.passages {
            passage_writer.add(passage)?;
        }
        passage_writer.finish()?;

        // Write ID mapping for integer-to-string ID lookup
        let ids_path = index_path.with_extension("ids.txt");
        let ids_content = self.ids.join("\n");
        std::fs::write(&ids_path, ids_content)?;

        // In recompute mode, save embeddings to a separate file
        if self.recompute_mode {
            let embeddings_path = index_path.with_extension("embeddings");
            let mut embeddings_writer = EmbeddingsWriter::create(&embeddings_path, self.dimensions)?;
            for embedding in &self.embeddings {
                embeddings_writer.add(embedding)?;
            }
            embeddings_writer.finish()?;
            info!("Embeddings saved to {:?}", embeddings_path);
        }

        // Build vector index using backend
        let backend = BackendBuilder::new(self.backend_type);
        backend.build(
            &self.embeddings,
            &self.ids,
            index_path,
            self.dimensions,
            self.graph_degree,
            self.complexity,
        )?;

        info!("Index built successfully at {:?}", index_path);
        Ok(())
    }
}
