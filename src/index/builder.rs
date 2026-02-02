//! Index builder - constructs vector indexes

use std::io::{BufWriter, Write};
use std::path::Path;

use tracing::info;

use crate::backend::{BackendBuilder, BackendType};

use super::embeddings::EmbeddingsWriter;
use super::passages::{Passage, PassageStore, PassageStoreWriter};

/// Builder for creating a LEANN index (in-memory, for small datasets)
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

/// Streaming builder that writes passages to disk as they come in,
/// only keeping embeddings in memory for final index build.
/// This reduces memory usage significantly for large datasets.
pub struct StreamingIndexBuilder {
    backend_type: BackendType,
    dimensions: usize,
    graph_degree: usize,
    complexity: usize,
    recompute_mode: bool,
    index_path: std::path::PathBuf,
    passage_writer: PassageStoreWriter,
    embeddings_writer: Option<EmbeddingsWriter>,
    ids_writer: BufWriter<std::fs::File>,
    /// Only store embeddings in memory for final HNSW build
    embeddings: Vec<Vec<f32>>,
    ids: Vec<String>,
    count: usize,
}

impl StreamingIndexBuilder {
    /// Create a streaming builder that writes to disk incrementally
    pub fn new(
        backend_type: BackendType,
        dimensions: usize,
        graph_degree: usize,
        complexity: usize,
        recompute_mode: bool,
        index_path: &Path,
    ) -> anyhow::Result<Self> {
        // Create parent directory
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let passage_writer = PassageStore::create(index_path)?;

        let embeddings_writer = if recompute_mode {
            let embeddings_path = index_path.with_extension("embeddings");
            Some(EmbeddingsWriter::create(&embeddings_path, dimensions)?)
        } else {
            None
        };

        let ids_path = index_path.with_extension("ids.txt");
        let ids_file = std::fs::File::create(&ids_path)?;
        let ids_writer = BufWriter::new(ids_file);

        Ok(Self {
            backend_type,
            dimensions,
            graph_degree,
            complexity,
            recompute_mode,
            index_path: index_path.to_path_buf(),
            passage_writer,
            embeddings_writer,
            ids_writer,
            embeddings: Vec::new(),
            ids: Vec::new(),
            count: 0,
        })
    }

    /// Add a passage - writes to disk immediately, keeps embedding in memory
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

        // Write passage to disk immediately
        let passage = Passage {
            id: id.to_string(),
            text: text.to_string(),
            metadata,
        };
        self.passage_writer.add(&passage)?;

        // Write ID to disk
        if self.count > 0 {
            writeln!(self.ids_writer)?;
        }
        write!(self.ids_writer, "{}", id)?;

        // Write embedding to disk if in recompute mode
        if let Some(ref mut writer) = self.embeddings_writer {
            writer.add(embedding)?;
        }

        // Keep embedding in memory for HNSW build (unfortunately required by usearch)
        self.embeddings.push(embedding.to_vec());
        self.ids.push(id.to_string());
        self.count += 1;

        Ok(())
    }

    /// Finalize and build the index
    pub fn build(mut self) -> anyhow::Result<()> {
        info!(
            "Building index with {} passages, {} dimensions{}",
            self.count,
            self.dimensions,
            if self.recompute_mode { " (recompute mode)" } else { "" }
        );

        // Flush all writers
        self.passage_writer.finish()?;
        self.ids_writer.flush()?;

        if let Some(writer) = self.embeddings_writer {
            writer.finish()?;
            info!("Embeddings saved to {:?}", self.index_path.with_extension("embeddings"));
        }

        // Build vector index using backend
        let backend = BackendBuilder::new(self.backend_type);
        backend.build(
            &self.embeddings,
            &self.ids,
            &self.index_path,
            self.dimensions,
            self.graph_degree,
            self.complexity,
        )?;

        info!("Index built successfully at {:?}", self.index_path);
        Ok(())
    }
}
