//! Backend module - vector search backends (HNSW, DiskANN)

mod traits;
mod hnsw;
mod compat;

#[cfg(feature = "diskann-backend")]
mod diskann;

pub use traits::{BackendBuilder, BackendSearcher};

use std::path::Path;

/// Supported backend types
#[derive(Debug, Clone, Copy)]
pub enum BackendType {
    Hnsw,
    DiskAnn,
}

impl BackendType {
    /// Load a searcher for this backend type
    pub fn load_searcher(
        self,
        index_path: &Path,
        dimensions: usize,
    ) -> anyhow::Result<Box<dyn BackendSearcher>> {
        match self {
            BackendType::Hnsw => {
                let searcher = hnsw::HnswSearcher::load(index_path, dimensions)?;
                Ok(Box::new(searcher))
            }
            #[cfg(feature = "diskann-backend")]
            BackendType::DiskAnn => {
                let searcher = diskann::DiskAnnSearcher::load(index_path, dimensions)?;
                Ok(Box::new(searcher))
            }
            #[cfg(not(feature = "diskann-backend"))]
            BackendType::DiskAnn => {
                anyhow::bail!(
                    "DiskANN backend not available. Rebuild with --features diskann-backend"
                )
            }
        }
    }
}

impl BackendBuilder {
    /// Create a new backend builder
    pub fn new(backend_type: BackendType) -> Self {
        Self { backend_type }
    }

    /// Build an index using the specified backend
    pub fn build(
        &self,
        embeddings: &[Vec<f32>],
        ids: &[String],
        index_path: &Path,
        dimensions: usize,
        graph_degree: usize,
        complexity: usize,
    ) -> anyhow::Result<()> {
        match self.backend_type {
            BackendType::Hnsw => {
                hnsw::build_index(embeddings, ids, index_path, dimensions, graph_degree, complexity)
            }
            #[cfg(feature = "diskann-backend")]
            BackendType::DiskAnn => {
                diskann::build_index(embeddings, ids, index_path, dimensions, graph_degree, complexity)
            }
            #[cfg(not(feature = "diskann-backend"))]
            BackendType::DiskAnn => {
                anyhow::bail!(
                    "DiskANN backend not available. Rebuild with --features diskann-backend"
                )
            }
        }
    }

    /// Add vectors to an existing index (HNSW only)
    pub fn add_to_index(
        &self,
        embeddings: &[Vec<f32>],
        index_path: &Path,
        dimensions: usize,
        start_id: usize,
    ) -> anyhow::Result<()> {
        match self.backend_type {
            BackendType::Hnsw => {
                hnsw::add_to_index(embeddings, index_path, dimensions, start_id)
            }
            BackendType::DiskAnn => {
                anyhow::bail!(
                    "DiskANN backend does not support incremental updates. \
                    Use --force to rebuild the entire index."
                )
            }
        }
    }
}
