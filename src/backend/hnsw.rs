//! HNSW backend using usearch crate

use std::path::Path;

use tracing::info;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use super::compat::is_faiss_index;
use super::traits::BackendSearcher;

/// HNSW searcher using usearch
pub struct HnswSearcher {
    index: Index,
}

impl HnswSearcher {
    /// Load an HNSW index from disk
    pub fn load(index_path: &Path, dimensions: usize) -> anyhow::Result<Self> {
        let index_file = index_path.with_extension("index");

        info!("Loading HNSW index from {:?}", index_file);

        // Check if this is a FAISS index (Python LEANN)
        if is_faiss_index(index_path) {
            anyhow::bail!(
                "This index was built with Python LEANN (FAISS format).\n\
                Rust LEANN uses usearch which has a different binary format.\n\n\
                To use this index with Rust LEANN, you need to rebuild it:\n\
                  leann build <name> --docs <path> --force\n\n\
                The passages and metadata files are compatible and will be preserved."
            );
        }

        if !index_file.exists() {
            anyhow::bail!(
                "Index file not found: {:?}\n\
                Run 'leann build' to create an index first.",
                index_file
            );
        }

        // Create index with same options used during build
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::IP, // Inner product (MIPS)
            quantization: ScalarKind::F32,
            connectivity: 32,
            expansion_add: 64,
            expansion_search: 64,
            multi: false,
        };

        let index = Index::new(&options)?;

        match index.load(index_file.to_string_lossy().as_ref()) {
            Ok(()) => {}
            Err(e) => {
                // Try to detect if it's a format mismatch
                let error_msg = format!("{}", e);
                if error_msg.contains("magic") || error_msg.contains("header") || error_msg.contains("version") {
                    anyhow::bail!(
                        "Failed to load index: incompatible format.\n\
                        This may be a FAISS index from Python LEANN.\n\
                        Rebuild with: leann build <name> --docs <path> --force\n\n\
                        Original error: {}", e
                    );
                }
                return Err(e.into());
            }
        }

        info!("Loaded HNSW index with {} vectors", index.size());

        Ok(Self { index })
    }
}

impl BackendSearcher for HnswSearcher {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _complexity: usize, // usearch doesn't expose search complexity
    ) -> anyhow::Result<(Vec<u64>, Vec<f32>)> {
        let results = self.index.search(query, top_k)?;

        Ok((results.keys.to_vec(), results.distances.to_vec()))
    }

    fn len(&self) -> usize {
        self.index.size()
    }
}

/// Build an HNSW index
pub fn build_index(
    embeddings: &[Vec<f32>],
    _ids: &[String],
    index_path: &Path,
    dimensions: usize,
    graph_degree: usize,
    complexity: usize,
) -> anyhow::Result<()> {
    info!(
        "Building HNSW index: {} vectors, {} dims, degree={}, complexity={}",
        embeddings.len(),
        dimensions,
        graph_degree,
        complexity
    );

    let options = IndexOptions {
        dimensions,
        metric: MetricKind::IP, // Inner product (MIPS)
        quantization: ScalarKind::F32,
        connectivity: graph_degree,
        expansion_add: complexity,
        expansion_search: complexity,
        multi: false,
    };

    let index = Index::new(&options)?;

    // Reserve capacity
    index.reserve(embeddings.len())?;

    // Add vectors
    for (i, embedding) in embeddings.iter().enumerate() {
        index.add(i as u64, embedding)?;
    }

    // Save to disk
    let index_file = index_path.with_extension("index");
    index.save(index_file.to_string_lossy().as_ref())?;

    info!("HNSW index saved to {:?}", index_file);

    Ok(())
}

/// Add vectors to an existing HNSW index
pub fn add_to_index(
    embeddings: &[Vec<f32>],
    index_path: &Path,
    dimensions: usize,
    start_id: usize,
) -> anyhow::Result<()> {
    let index_file = index_path.with_extension("index");

    info!(
        "Adding {} vectors to HNSW index at {:?}",
        embeddings.len(),
        index_file
    );

    // Load existing index
    let options = IndexOptions {
        dimensions,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 32,
        expansion_add: 64,
        expansion_search: 64,
        multi: false,
    };

    let index = Index::new(&options)?;
    index.load(index_file.to_string_lossy().as_ref())?;

    let current_size = index.size();
    info!("Loaded index with {} existing vectors", current_size);

    // Reserve additional capacity
    index.reserve(current_size + embeddings.len())?;

    // Add new vectors
    for (i, embedding) in embeddings.iter().enumerate() {
        let id = (start_id + i) as u64;
        index.add(id, embedding)?;
    }

    // Save updated index
    index.save(index_file.to_string_lossy().as_ref())?;

    info!(
        "Updated index saved with {} total vectors",
        index.size()
    );

    Ok(())
}
