//! Python index compatibility
//!
//! Detects Python LEANN indexes (which use FAISS) and provides guidance
//! on rebuilding with usearch.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Check if an index file is a FAISS format (Python LEANN)
pub fn is_faiss_index(index_path: &Path) -> bool {
    let index_file = index_path.with_extension("index");

    if !index_file.exists() {
        return false;
    }

    // FAISS index files start with a magic header
    // Common FAISS signatures include "IxFl" (IndexFlat), "IxHN" (IndexHNSW), etc.
    if let Ok(file) = File::open(&index_file) {
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 4];
        if std::io::Read::read_exact(&mut reader, &mut header).is_ok() {
            // Check for common FAISS magic bytes
            // FAISS uses various 4-byte headers like "IxFl", "IxHN", "IxIV", etc.
            if header[0] == b'I' && header[1] == b'x' {
                return true;
            }
            // Also check for FAISS CSR format (compact HNSW)
            if &header == b"CSR\x00" || &header == b"HNSW" {
                return true;
            }
        }
    }

    false
}

/// Check if an index is a usearch format (Rust LEANN)
pub fn is_usearch_index(index_path: &Path) -> bool {
    let index_file = index_path.with_extension("index");

    if !index_file.exists() {
        return false;
    }

    // usearch indexes have a different magic header
    if let Ok(file) = File::open(&index_file) {
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 8];
        if std::io::Read::read_exact(&mut reader, &mut header).is_ok() {
            // usearch v2 magic: first 4 bytes are version info
            // This is a heuristic - usearch doesn't have a documented magic number
            // If it's not FAISS, assume it might be usearch
            if !(header[0] == b'I' && header[1] == b'x') {
                return true;
            }
        }
    }

    false
}

/// Get index compatibility info
pub fn get_index_info(index_path: &Path) -> IndexInfo {
    let index_file = index_path.with_extension("index");
    let ids_file = index_path.with_extension("ids.txt");
    let passages_file = index_path.with_extension("passages.jsonl");
    let idx_json_file = index_path.with_extension("passages.idx.json");
    let idx_pickle_file = index_path.with_extension("idx");

    IndexInfo {
        has_vector_index: index_file.exists(),
        is_faiss: is_faiss_index(index_path),
        is_usearch: is_usearch_index(index_path),
        has_id_map: ids_file.exists(),
        has_passages: passages_file.exists(),
        has_json_offset: idx_json_file.exists(),
        has_pickle_offset: idx_pickle_file.exists(),
    }
}

/// Index compatibility information
#[derive(Debug)]
pub struct IndexInfo {
    pub has_vector_index: bool,
    pub is_faiss: bool,
    pub is_usearch: bool,
    pub has_id_map: bool,
    pub has_passages: bool,
    pub has_json_offset: bool,
    pub has_pickle_offset: bool,
}

impl IndexInfo {
    /// Check if this is a Python LEANN index
    pub fn is_python_index(&self) -> bool {
        self.is_faiss || self.has_pickle_offset
    }

    /// Check if this index can be read by Rust LEANN
    pub fn is_rust_compatible(&self) -> bool {
        self.is_usearch && self.has_json_offset && self.has_passages
    }

    /// Get rebuild message if needed
    pub fn rebuild_message(&self) -> Option<String> {
        if self.is_faiss {
            Some(
                "This index was built with Python LEANN (FAISS format). \
                Rust LEANN uses usearch which has a different format. \
                To use this index, rebuild it with: \
                leann build <name> --docs <path> --force".to_string()
            )
        } else if self.has_pickle_offset && !self.has_json_offset {
            Some(
                "This index uses Python pickle offset format (.idx). \
                Rust LEANN requires JSON offset format (.passages.idx.json). \
                Please rebuild the index: leann build <name> --docs <path> --force".to_string()
            )
        } else {
            None
        }
    }
}
