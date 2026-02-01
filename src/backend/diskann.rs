//! DiskANN backend using diskann-rs crate
//!
//! Provides disk-based approximate nearest neighbor search using the Vamana graph algorithm.
//! Memory-efficient for larger-than-RAM datasets.

use std::path::Path;

use anndists::dist::distances::DistDot;
use diskann_rs::{DiskANN, DiskAnnParams};
use tracing::info;

use super::traits::BackendSearcher;

/// DiskANN searcher using diskann-rs
pub struct DiskAnnSearcher {
    index: DiskANN<DistDot>,
}

impl DiskAnnSearcher {
    /// Load a DiskANN index from disk
    pub fn load(index_path: &Path, _dimensions: usize) -> anyhow::Result<Self> {
        let index_file = index_path.with_extension("diskann");

        info!("Loading DiskANN index from {:?}", index_file);

        if !index_file.exists() {
            anyhow::bail!(
                "DiskANN index not found: {:?}\n\
                Run 'leann build' with --backend-name diskann to create an index first.",
                index_file
            );
        }

        let index = DiskANN::<DistDot>::open_index_with(
            index_file.to_string_lossy().as_ref(),
            DistDot {},
        )
        .map_err(|e| anyhow::anyhow!("Failed to load DiskANN index: {}", e))?;

        info!("Loaded DiskANN index with {} vectors", index.num_vectors);

        Ok(Self { index })
    }
}

impl BackendSearcher for DiskAnnSearcher {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        complexity: usize,
    ) -> anyhow::Result<(Vec<u64>, Vec<f32>)> {
        // Use complexity as beam_width for search
        let beam_width = complexity.max(top_k);

        let results = self.index.search_with_dists(query, top_k, beam_width);

        let indices: Vec<u64> = results.iter().map(|(id, _)| *id as u64).collect();
        let distances: Vec<f32> = results.iter().map(|(_, d)| *d).collect();

        Ok((indices, distances))
    }

    fn len(&self) -> usize {
        self.index.num_vectors
    }
}

/// Build a DiskANN index
pub fn build_index(
    embeddings: &[Vec<f32>],
    _ids: &[String],
    index_path: &Path,
    dimensions: usize,
    graph_degree: usize,
    complexity: usize,
) -> anyhow::Result<()> {
    info!(
        "Building DiskANN index: {} vectors, {} dims, degree={}, complexity={}",
        embeddings.len(),
        dimensions,
        graph_degree,
        complexity
    );

    let index_file = index_path.with_extension("diskann");

    let params = DiskAnnParams {
        max_degree: graph_degree,
        build_beam_width: complexity,
        alpha: 1.2,
    };

    let _index = DiskANN::<DistDot>::build_index_with_params(
        embeddings,
        DistDot {},
        index_file.to_string_lossy().as_ref(),
        params,
    )
    .map_err(|e| anyhow::anyhow!("Failed to build DiskANN index: {}", e))?;

    info!("DiskANN index saved to {:?}", index_file);

    Ok(())
}
