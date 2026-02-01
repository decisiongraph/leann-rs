//! Embeddings storage - memory-mapped embedding vectors
//!
//! Stores embeddings separately from the vector index for recomputation support.
//! Can be deleted to "prune" the index and enable on-demand recomputation.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

/// Embeddings storage using memory-mapped file
pub struct EmbeddingsStore {
    mmap: Mmap,
    dimensions: usize,
    count: usize,
}

impl EmbeddingsStore {
    /// Open an existing embeddings file
    pub fn open(path: &Path, dimensions: usize) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Calculate count from file size
        let bytes_per_embedding = dimensions * std::mem::size_of::<f32>();
        let count = mmap.len() / bytes_per_embedding;

        Ok(Self {
            mmap,
            dimensions,
            count,
        })
    }

    /// Check if embeddings file exists
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    /// Get the path for embeddings file given index path
    pub fn path_for_index(index_path: &Path) -> std::path::PathBuf {
        index_path.with_extension("embeddings")
    }

    /// Get embedding for a specific index
    pub fn get(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.count {
            return None;
        }

        let bytes_per_embedding = self.dimensions * std::mem::size_of::<f32>();
        let start = idx * bytes_per_embedding;
        let end = start + bytes_per_embedding;

        if end > self.mmap.len() {
            return None;
        }

        // Safety: We're reading f32 values that were written as f32
        let slice = &self.mmap[start..end];
        let ptr = slice.as_ptr() as *const f32;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.dimensions) })
    }

    /// Get all embeddings as a vector of slices
    pub fn get_all(&self) -> Vec<&[f32]> {
        (0..self.count).filter_map(|i| self.get(i)).collect()
    }

    /// Get number of embeddings
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Writer for creating embeddings storage
pub struct EmbeddingsWriter {
    writer: BufWriter<File>,
    dimensions: usize,
    count: usize,
}

impl EmbeddingsWriter {
    /// Create a new embeddings file
    pub fn create(path: &Path, dimensions: usize) -> anyhow::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            dimensions,
            count: 0,
        })
    }

    /// Add an embedding
    pub fn add(&mut self, embedding: &[f32]) -> anyhow::Result<()> {
        if embedding.len() != self.dimensions {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            );
        }

        // Write embedding as raw bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            )
        };
        self.writer.write_all(bytes)?;
        self.count += 1;

        Ok(())
    }

    /// Finish writing
    pub fn finish(mut self) -> anyhow::Result<usize> {
        self.writer.flush()?;
        Ok(self.count)
    }

    /// Get current count
    pub fn len(&self) -> usize {
        self.count
    }
}

/// Delete embeddings file to enable recomputation mode
pub fn prune_embeddings(index_path: &Path) -> anyhow::Result<()> {
    let embeddings_path = EmbeddingsStore::path_for_index(index_path);
    if embeddings_path.exists() {
        std::fs::remove_file(&embeddings_path)?;
    }
    Ok(())
}
