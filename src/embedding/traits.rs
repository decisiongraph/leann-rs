//! Embedding provider traits

use async_trait::async_trait;

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProviderTrait: Send + Sync {
    /// Get embedding dimensions
    fn dimensions(&self) -> usize;

    /// Compute embeddings for texts
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}
