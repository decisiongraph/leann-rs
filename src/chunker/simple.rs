//! Simple character-based text chunking

use std::path::Path;

use super::{Chunk, Chunker};

/// Simple chunker that splits by character count with word boundary awareness
pub struct SimpleChunker {
    /// Approximate chunk size in tokens (1 token ~= 4 chars)
    chunk_size: usize,
    /// Overlap size in tokens
    chunk_overlap: usize,
}

impl SimpleChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl Chunker for SimpleChunker {
    fn chunk(
        &self,
        text: &str,
        source_path: &Path,
        chunk_id: &mut u64,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();

        // Simple approximation: 1 token ~= 4 characters
        let char_chunk_size = self.chunk_size * 4;
        let char_overlap = self.chunk_overlap * 4;

        if text.len() <= char_chunk_size {
            *chunk_id += 1;
            chunks.push(Chunk {
                id: chunk_id.to_string(),
                text: text.to_string(),
                metadata: serde_json::json!({
                    "source": source_path.to_string_lossy(),
                    "chunk_index": 0,
                    "chunk_type": "simple",
                }),
            });
            return chunks;
        }

        let mut start = 0;
        let mut chunk_index = 0;

        while start < text.len() {
            let end = (start + char_chunk_size).min(text.len());

            // Try to break at word boundary
            let chunk_end = if end < text.len() {
                text[start..end]
                    .rfind(|c: char| c.is_whitespace())
                    .map(|pos| start + pos)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_text = text[start..chunk_end].trim().to_string();

            if !chunk_text.is_empty() {
                *chunk_id += 1;
                chunks.push(Chunk {
                    id: chunk_id.to_string(),
                    text: chunk_text,
                    metadata: serde_json::json!({
                        "source": source_path.to_string_lossy(),
                        "chunk_index": chunk_index,
                        "chunk_type": "simple",
                    }),
                });
                chunk_index += 1;
            }

            // Move start with overlap
            start = if chunk_end > start + char_overlap {
                chunk_end - char_overlap
            } else {
                chunk_end
            };

            // Prevent infinite loop
            if start >= text.len() || chunk_end >= text.len() {
                break;
            }
        }

        chunks
    }
}
