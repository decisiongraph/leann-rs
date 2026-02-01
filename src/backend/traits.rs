//! Backend traits for vector search

use super::BackendType;

/// Builder for creating vector indexes
pub struct BackendBuilder {
    pub(crate) backend_type: BackendType,
}

/// Trait for searching a vector index
pub trait BackendSearcher: Send + Sync {
    /// Search for nearest neighbors
    ///
    /// Returns (indices, distances) where indices are integer offsets
    /// into the original embedding order.
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        complexity: usize,
    ) -> anyhow::Result<(Vec<u64>, Vec<f32>)>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
