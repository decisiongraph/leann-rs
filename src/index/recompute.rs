//! Recompute searcher - brute-force search with on-demand embedding computation
//!
//! Used when an index has been pruned (embeddings deleted to save space).
//! Recomputes embeddings for all passages during search.

use std::path::Path;

use tracing::info;

use crate::embedding::EmbeddingProvider;

use super::filter::MetadataFilter;
use super::passages::PassageStore;
use super::searcher::SearchResult;

/// Searcher that recomputes embeddings on-demand (for pruned indices)
pub struct RecomputeSearcher {
    passages: PassageStore,
    id_map: Vec<String>,
    dimensions: usize,
}

impl RecomputeSearcher {
    /// Load a pruned index for recompute search
    pub fn load(index_path: &Path, dimensions: usize) -> anyhow::Result<Self> {
        info!("Loading pruned index for recompute search from {:?}", index_path);

        // Load passages
        let passages = PassageStore::open(index_path)?;

        // Load ID mapping
        let ids_path = index_path.with_extension("ids.txt");
        let id_map: Vec<String> = if ids_path.exists() {
            std::fs::read_to_string(&ids_path)?
                .lines()
                .map(|s| s.to_string())
                .collect()
        } else {
            passages.ids().cloned().collect()
        };

        info!("Loaded {} passages for recompute search", id_map.len());

        Ok(Self {
            passages,
            id_map,
            dimensions,
        })
    }

    /// Search using brute-force with on-demand embedding computation
    pub async fn search(
        &self,
        query_embedding: &[f32],
        embedding_provider: &EmbeddingProvider,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        info!("Recompute search: computing embeddings for {} passages", self.id_map.len());

        // Collect all passage texts
        let mut texts: Vec<String> = Vec::with_capacity(self.id_map.len());
        let mut valid_indices: Vec<usize> = Vec::with_capacity(self.id_map.len());

        for (idx, id) in self.id_map.iter().enumerate() {
            match self.passages.get(id) {
                Ok(passage) => {
                    // Apply filter early to avoid computing unnecessary embeddings
                    if let Some(f) = filter {
                        if !f.matches(&passage.metadata) {
                            continue;
                        }
                    }
                    texts.push(passage.text);
                    valid_indices.push(idx);
                }
                Err(_) => continue,
            }
        }

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Compute embeddings in batches
        let batch_size = 100;
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
            let embeddings = embedding_provider.embed(&batch_refs).await?;
            all_embeddings.extend(embeddings);
        }

        // Compute distances (inner product for MIPS)
        let mut scored: Vec<(usize, f32)> = all_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let score = dot_product(query_embedding, emb);
                (valid_indices[i], score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k and build results
        let mut results = Vec::with_capacity(top_k);
        for (idx, score) in scored.into_iter().take(top_k) {
            let id = &self.id_map[idx];
            if let Ok(passage) = self.passages.get(id) {
                results.push(SearchResult {
                    id: id.clone(),
                    score,
                    text: passage.text,
                    metadata: passage.metadata,
                });
            }
        }

        Ok(results)
    }

    /// Get passage count
    pub fn len(&self) -> usize {
        self.id_map.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.id_map.is_empty()
    }
}

/// Compute dot product (inner product) between two vectors
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
