//! Chunking module - text chunking strategies for indexing
//!
//! Provides both simple character-based chunking and AST-aware code chunking.

mod ast;
mod simple;

pub use ast::CodeChunker;
pub use simple::SimpleChunker;

use std::path::Path;

/// A text chunk with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub metadata: serde_json::Value,
}

/// Chunking strategy to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    /// Simple character-based chunking
    Simple,
    /// AST-aware chunking for code files
    Ast,
    /// Auto-detect based on file type
    Auto,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for ChunkingStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(Self::Simple),
            "ast" => Ok(Self::Ast),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("Unknown chunking strategy: {}", s)),
        }
    }
}

/// Trait for chunkers
pub trait Chunker {
    /// Chunk text content into smaller pieces
    fn chunk(
        &self,
        text: &str,
        source_path: &Path,
        chunk_id: &mut u64,
    ) -> Vec<Chunk>;
}

/// Unified chunker that selects strategy based on file type
pub struct SmartChunker {
    strategy: ChunkingStrategy,
    simple: SimpleChunker,
    ast: CodeChunker,
}

impl SmartChunker {
    pub fn new(
        strategy: ChunkingStrategy,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Self {
        Self {
            strategy,
            simple: SimpleChunker::new(chunk_size, chunk_overlap),
            ast: CodeChunker::new(chunk_size, chunk_overlap),
        }
    }

    /// Check if a file should use AST chunking
    fn should_use_ast(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        matches!(
            ext,
            "rs" | "py" | "js" | "ts" | "tsx" | "jsx" |
            "go" | "java" | "c" | "cpp" | "cc" | "h" | "hpp" |
            "rb" | "php" | "swift" | "kt" | "scala" | "cs"
        )
    }
}

impl Chunker for SmartChunker {
    fn chunk(
        &self,
        text: &str,
        source_path: &Path,
        chunk_id: &mut u64,
    ) -> Vec<Chunk> {
        match self.strategy {
            ChunkingStrategy::Simple => self.simple.chunk(text, source_path, chunk_id),
            ChunkingStrategy::Ast => self.ast.chunk(text, source_path, chunk_id),
            ChunkingStrategy::Auto => {
                if self.should_use_ast(source_path) {
                    self.ast.chunk(text, source_path, chunk_id)
                } else {
                    self.simple.chunk(text, source_path, chunk_id)
                }
            }
        }
    }
}
