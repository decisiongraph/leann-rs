//! Passage storage - JSONL format with offset index

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A single passage with text and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Passage store using JSONL file with JSON offset index
pub struct PassageStore {
    /// Offset map: passage_id -> byte offset in JSONL file
    offsets: HashMap<String, u64>,

    /// Path to JSONL file
    jsonl_path: std::path::PathBuf,
}

impl PassageStore {
    /// Create a new passage store for writing
    pub fn create(base_path: &Path) -> anyhow::Result<PassageStoreWriter> {
        let jsonl_path = base_path.with_extension("passages.jsonl");
        let idx_path = base_path.with_extension("passages.idx.json");

        let file = File::create(&jsonl_path)?;
        let writer = BufWriter::new(file);

        Ok(PassageStoreWriter {
            writer,
            offsets: HashMap::new(),
            jsonl_path,
            idx_path,
            current_offset: 0,
        })
    }

    /// Open an existing passage store for reading
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        let jsonl_path = base_path.with_extension("passages.jsonl");
        let idx_path = base_path.with_extension("passages.idx.json");

        // Load offset index
        let idx_content = std::fs::read_to_string(&idx_path)?;
        let offsets: HashMap<String, u64> = serde_json::from_str(&idx_content)?;

        Ok(Self {
            offsets,
            jsonl_path,
        })
    }

    /// Open an existing passage store for appending
    pub fn open_for_append(base_path: &Path) -> anyhow::Result<PassageStoreWriter> {
        let jsonl_path = base_path.with_extension("passages.jsonl");
        let idx_path = base_path.with_extension("passages.idx.json");

        // Load existing offset index
        let idx_content = std::fs::read_to_string(&idx_path)?;
        let offsets: HashMap<String, u64> = serde_json::from_str(&idx_content)?;

        // Open file for appending
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(&jsonl_path)?;

        // Get current file size for offset tracking
        let current_offset = file.metadata()?.len();

        let writer = BufWriter::new(file);

        Ok(PassageStoreWriter {
            writer,
            offsets,
            jsonl_path,
            idx_path,
            current_offset,
        })
    }

    /// Get a passage by ID
    pub fn get(&self, id: &str) -> anyhow::Result<Passage> {
        let offset = self
            .offsets
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("Passage not found: {}", id))?;

        let mut file = File::open(&self.jsonl_path)?;
        file.seek(SeekFrom::Start(*offset))?;

        let mut reader = BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line)?;

        let passage: Passage = serde_json::from_str(&line)?;
        Ok(passage)
    }

    /// Get all passage IDs
    pub fn ids(&self) -> impl Iterator<Item = &String> {
        self.offsets.keys()
    }

    /// Get passage count
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Writer for creating a passage store
pub struct PassageStoreWriter {
    writer: BufWriter<File>,
    offsets: HashMap<String, u64>,
    jsonl_path: std::path::PathBuf,
    idx_path: std::path::PathBuf,
    current_offset: u64,
}

impl PassageStoreWriter {
    /// Add a passage to the store
    pub fn add(&mut self, passage: &Passage) -> anyhow::Result<()> {
        // Record offset before writing
        self.offsets.insert(passage.id.clone(), self.current_offset);

        // Write JSON line
        let json = serde_json::to_string(passage)?;
        self.writer.write_all(json.as_bytes())?;
        self.writer.write_all(b"\n")?;

        // Update offset
        self.current_offset += json.len() as u64 + 1; // +1 for newline

        Ok(())
    }

    /// Finish writing and save the offset index
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.writer.flush()?;

        // Save offset index as JSON (not pickle, for cross-platform compatibility)
        let idx_content = serde_json::to_string(&self.offsets)?;
        std::fs::write(&self.idx_path, idx_content)?;

        Ok(())
    }

    /// Get the JSONL path
    pub fn jsonl_path(&self) -> &Path {
        &self.jsonl_path
    }

    /// Get current passage count
    pub fn len(&self) -> usize {
        self.offsets.len()
    }
}
