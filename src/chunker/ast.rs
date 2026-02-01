//! AST-aware code chunking
//!
//! Uses regex patterns to identify semantic code units (functions, classes, etc.)
//! and chunks code by these boundaries rather than arbitrary character positions.

use std::path::Path;

use regex::Regex;

use super::{Chunk, Chunker};

/// Code block detected by pattern matching
#[derive(Debug)]
struct CodeBlock {
    /// Type of block (function, class, struct, etc.)
    block_type: String,
    /// Name of the block (function/class name)
    name: String,
    /// Start line (0-indexed)
    start_line: usize,
    /// End line (0-indexed, exclusive)
    end_line: usize,
    /// The actual text content
    content: String,
}

/// AST-aware code chunker
pub struct CodeChunker {
    /// Maximum chunk size in tokens
    max_chunk_size: usize,
    /// Overlap size in tokens
    chunk_overlap: usize,
}

impl CodeChunker {
    pub fn new(max_chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            max_chunk_size,
            chunk_overlap,
        }
    }

    /// Get language from file extension
    fn get_language(&self, path: &Path) -> Option<&'static str> {
        let ext = path.extension()?.to_str()?;
        match ext {
            "rs" => Some("rust"),
            "py" => Some("python"),
            "js" | "jsx" => Some("javascript"),
            "ts" | "tsx" => Some("typescript"),
            "go" => Some("go"),
            "java" => Some("java"),
            "c" | "h" => Some("c"),
            "cpp" | "cc" | "hpp" => Some("cpp"),
            "rb" => Some("ruby"),
            "php" => Some("php"),
            "swift" => Some("swift"),
            "kt" => Some("kotlin"),
            "scala" => Some("scala"),
            "cs" => Some("csharp"),
            _ => None,
        }
    }

    /// Extract code blocks from source
    fn extract_blocks(&self, text: &str, language: &str) -> Vec<CodeBlock> {
        let lines: Vec<&str> = text.lines().collect();
        let mut blocks = Vec::new();

        // Get patterns for this language
        let patterns = self.get_patterns(language);

        for pattern in patterns {
            let regex = match Regex::new(pattern.regex) {
                Ok(r) => r,
                Err(_) => continue,
            };

            for (line_idx, line) in lines.iter().enumerate() {
                if let Some(caps) = regex.captures(line) {
                    let name = caps.get(1).map(|m| m.as_str()).unwrap_or("anonymous");

                    // Find the end of this block by tracking brace/indentation
                    let end_line = self.find_block_end(&lines, line_idx, language);

                    let content: String = lines[line_idx..end_line]
                        .iter()
                        .map(|s| *s)
                        .collect::<Vec<_>>()
                        .join("\n");

                    blocks.push(CodeBlock {
                        block_type: pattern.block_type.to_string(),
                        name: name.to_string(),
                        start_line: line_idx,
                        end_line,
                        content,
                    });
                }
            }
        }

        // Sort by start line and remove overlapping blocks (keep larger ones)
        blocks.sort_by_key(|b| b.start_line);
        self.deduplicate_blocks(blocks)
    }

    /// Find the end of a code block
    fn find_block_end(&self, lines: &[&str], start: usize, language: &str) -> usize {
        match language {
            "python" | "ruby" => self.find_indentation_end(lines, start),
            _ => self.find_brace_end(lines, start),
        }
    }

    /// Find end of indentation-based block (Python, Ruby)
    fn find_indentation_end(&self, lines: &[&str], start: usize) -> usize {
        if start >= lines.len() {
            return start + 1;
        }

        let base_indent = lines[start].len() - lines[start].trim_start().len();
        let mut end = start + 1;

        while end < lines.len() {
            let line = lines[end];

            // Skip empty lines
            if line.trim().is_empty() {
                end += 1;
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            // If indentation decreases or equals base (and is not empty), block ends
            if indent <= base_indent {
                break;
            }

            end += 1;
        }

        end
    }

    /// Find end of brace-based block (C-like languages)
    fn find_brace_end(&self, lines: &[&str], start: usize) -> usize {
        let mut brace_count = 0;
        let mut found_first_brace = false;

        for (idx, line) in lines.iter().enumerate().skip(start) {
            for ch in line.chars() {
                match ch {
                    '{' => {
                        brace_count += 1;
                        found_first_brace = true;
                    }
                    '}' => {
                        brace_count -= 1;
                        if found_first_brace && brace_count == 0 {
                            return idx + 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Fallback: return a reasonable chunk
        (start + 50).min(lines.len())
    }

    /// Remove overlapping blocks, keeping larger ones
    fn deduplicate_blocks(&self, blocks: Vec<CodeBlock>) -> Vec<CodeBlock> {
        let mut result = Vec::new();

        for block in blocks {
            // Check if this block overlaps with any existing block
            let overlaps = result.iter().any(|existing: &CodeBlock| {
                block.start_line < existing.end_line && block.end_line > existing.start_line
            });

            if !overlaps {
                result.push(block);
            }
        }

        result
    }

    /// Get regex patterns for detecting code blocks
    fn get_patterns(&self, language: &str) -> Vec<Pattern> {
        match language {
            "rust" => vec![
                Pattern::new("function", r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"),
                Pattern::new("struct", r"^\s*(?:pub\s+)?struct\s+(\w+)"),
                Pattern::new("enum", r"^\s*(?:pub\s+)?enum\s+(\w+)"),
                Pattern::new("impl", r"^\s*impl(?:<[^>]+>)?\s+(?:(\w+)|for\s+(\w+))"),
                Pattern::new("trait", r"^\s*(?:pub\s+)?trait\s+(\w+)"),
                Pattern::new("mod", r"^\s*(?:pub\s+)?mod\s+(\w+)\s*\{"),
            ],
            "python" => vec![
                Pattern::new("function", r"^\s*(?:async\s+)?def\s+(\w+)"),
                Pattern::new("class", r"^\s*class\s+(\w+)"),
            ],
            "javascript" | "typescript" => vec![
                Pattern::new("function", r"^\s*(?:async\s+)?function\s+(\w+)"),
                Pattern::new("function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"),
                Pattern::new("class", r"^\s*(?:export\s+)?class\s+(\w+)"),
                Pattern::new("method", r"^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"),
                Pattern::new("arrow", r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"),
            ],
            "go" => vec![
                Pattern::new("function", r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)"),
                Pattern::new("struct", r"^\s*type\s+(\w+)\s+struct"),
                Pattern::new("interface", r"^\s*type\s+(\w+)\s+interface"),
            ],
            "java" => vec![
                Pattern::new("class", r"^\s*(?:public\s+)?(?:abstract\s+)?class\s+(\w+)"),
                Pattern::new("interface", r"^\s*(?:public\s+)?interface\s+(\w+)"),
                Pattern::new("method", r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\("),
            ],
            "c" | "cpp" => vec![
                Pattern::new("function", r"^\s*(?:\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*\{?"),
                Pattern::new("class", r"^\s*class\s+(\w+)"),
                Pattern::new("struct", r"^\s*struct\s+(\w+)"),
            ],
            "ruby" => vec![
                Pattern::new("class", r"^\s*class\s+(\w+)"),
                Pattern::new("module", r"^\s*module\s+(\w+)"),
                Pattern::new("method", r"^\s*def\s+(\w+)"),
            ],
            _ => vec![
                // Generic patterns for unknown languages
                Pattern::new("function", r"^\s*(?:def|func|function)\s+(\w+)"),
                Pattern::new("class", r"^\s*class\s+(\w+)"),
            ],
        }
    }

    /// Create chunks from blocks, respecting size limits
    fn blocks_to_chunks(
        &self,
        blocks: &[CodeBlock],
        source_path: &Path,
        chunk_id: &mut u64,
        language: &str,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let char_limit = self.max_chunk_size * 4; // ~4 chars per token

        for block in blocks {
            if block.content.len() <= char_limit {
                // Block fits in one chunk
                *chunk_id += 1;
                chunks.push(Chunk {
                    id: chunk_id.to_string(),
                    text: block.content.clone(),
                    metadata: serde_json::json!({
                        "source": source_path.to_string_lossy(),
                        "chunk_type": "ast",
                        "block_type": block.block_type,
                        "name": block.name,
                        "language": language,
                        "start_line": block.start_line + 1,
                        "end_line": block.end_line,
                    }),
                });
            } else {
                // Block is too large, split by lines
                let lines: Vec<&str> = block.content.lines().collect();
                let mut line_idx = 0;
                let mut part = 0;

                while line_idx < lines.len() {
                    let mut chunk_lines = Vec::new();
                    let mut chunk_len = 0;

                    while line_idx < lines.len() && chunk_len < char_limit {
                        let line = lines[line_idx];
                        chunk_len += line.len() + 1; // +1 for newline
                        chunk_lines.push(line);
                        line_idx += 1;
                    }

                    // Add overlap
                    let overlap_lines = (self.chunk_overlap * 4) / 80; // ~80 chars per line
                    line_idx = line_idx.saturating_sub(overlap_lines);

                    if !chunk_lines.is_empty() {
                        *chunk_id += 1;
                        chunks.push(Chunk {
                            id: chunk_id.to_string(),
                            text: chunk_lines.join("\n"),
                            metadata: serde_json::json!({
                                "source": source_path.to_string_lossy(),
                                "chunk_type": "ast",
                                "block_type": block.block_type,
                                "name": format!("{}_part{}", block.name, part),
                                "language": language,
                                "start_line": block.start_line + 1,
                                "end_line": block.end_line,
                                "part": part,
                            }),
                        });
                        part += 1;
                    }
                }
            }
        }

        chunks
    }

    /// Fill gaps between blocks with simple chunks
    fn fill_gaps(
        &self,
        lines: &[&str],
        blocks: &[CodeBlock],
        source_path: &Path,
        chunk_id: &mut u64,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let char_limit = self.max_chunk_size * 4;

        let mut current_line = 0;

        for block in blocks {
            if block.start_line > current_line {
                // There's a gap before this block
                let gap_text: String = lines[current_line..block.start_line]
                    .iter()
                    .map(|s| *s)
                    .collect::<Vec<_>>()
                    .join("\n");

                // Only create chunk if gap is non-trivial
                let trimmed = gap_text.trim();
                if !trimmed.is_empty() && trimmed.len() > 20 {
                    // Split gap if too large
                    if gap_text.len() <= char_limit {
                        *chunk_id += 1;
                        chunks.push(Chunk {
                            id: chunk_id.to_string(),
                            text: gap_text,
                            metadata: serde_json::json!({
                                "source": source_path.to_string_lossy(),
                                "chunk_type": "context",
                                "start_line": current_line + 1,
                                "end_line": block.start_line,
                            }),
                        });
                    }
                    // If gap is too large, we just skip it (imports, etc.)
                }
            }
            current_line = block.end_line;
        }

        chunks
    }
}

impl Chunker for CodeChunker {
    fn chunk(
        &self,
        text: &str,
        source_path: &Path,
        chunk_id: &mut u64,
    ) -> Vec<Chunk> {
        let language = match self.get_language(source_path) {
            Some(lang) => lang,
            None => {
                // Fallback to simple chunking for unknown languages
                let simple = super::SimpleChunker::new(self.max_chunk_size, self.chunk_overlap);
                return simple.chunk(text, source_path, chunk_id);
            }
        };

        let blocks = self.extract_blocks(text, language);

        if blocks.is_empty() {
            // No recognizable blocks, fallback to simple chunking
            let simple = super::SimpleChunker::new(self.max_chunk_size, self.chunk_overlap);
            return simple.chunk(text, source_path, chunk_id);
        }

        let lines: Vec<&str> = text.lines().collect();

        // Create chunks from blocks
        let mut chunks = self.blocks_to_chunks(&blocks, source_path, chunk_id, language);

        // Fill gaps with context chunks
        let gap_chunks = self.fill_gaps(&lines, &blocks, source_path, chunk_id);

        // Merge and sort by position
        chunks.extend(gap_chunks);
        chunks.sort_by(|a, b| {
            let a_line: usize = a.metadata["start_line"].as_u64().unwrap_or(0) as usize;
            let b_line: usize = b.metadata["start_line"].as_u64().unwrap_or(0) as usize;
            a_line.cmp(&b_line)
        });

        chunks
    }
}

/// Pattern for matching code blocks
struct Pattern {
    block_type: &'static str,
    regex: &'static str,
}

impl Pattern {
    fn new(block_type: &'static str, regex: &'static str) -> Self {
        Self { block_type, regex }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_detection() {
        let chunker = CodeChunker::new(256, 64);
        let code = r#"
fn hello_world() {
    println!("Hello, world!");
}

pub async fn async_func() -> Result<(), Error> {
    Ok(())
}
"#;
        let mut chunk_id = 0;
        let chunks = chunker.chunk(code, Path::new("test.rs"), &mut chunk_id);

        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.metadata["name"]
            .as_str()
            .map(|s| s.contains("hello"))
            .unwrap_or(false)));
    }

    #[test]
    fn test_python_class_detection() {
        let chunker = CodeChunker::new(256, 64);
        let code = r#"
class MyClass:
    def __init__(self):
        self.value = 0

    def get_value(self):
        return self.value

def standalone_func():
    pass
"#;
        let mut chunk_id = 0;
        let chunks = chunker.chunk(code, Path::new("test.py"), &mut chunk_id);

        assert!(!chunks.is_empty());
    }
}
