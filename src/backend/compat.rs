//! Python index compatibility
//!
//! Detects Python LEANN indexes (which use FAISS) and provides guidance
//! on rebuilding with usearch.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Check if an index file is a FAISS format (Python LEANN)
///
/// FAISS indexes use magic headers like "IxFl" (IndexFlat), "IxHN" (IndexHNSW), etc.
/// This detection helps provide useful error messages when users try to load
/// Python LEANN indexes with the Rust version.
pub fn is_faiss_index(index_path: &Path) -> bool {
    let index_file = index_path.with_extension("index");

    if !index_file.exists() {
        return false;
    }

    if let Ok(file) = File::open(&index_file) {
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 4];
        if std::io::Read::read_exact(&mut reader, &mut header).is_ok() {
            // Check for common FAISS magic bytes
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
