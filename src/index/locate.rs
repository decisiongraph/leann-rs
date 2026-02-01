//! Index location utilities

use std::path::PathBuf;

/// Find an index by name in the current project or global registry
///
/// Search order:
/// 1. Local project: `.leann/indexes/<name>`
/// 2. Absolute path (if provided)
/// 3. Global user registry: `~/.leann/indexes/<name>`
pub fn find_index(name: &str) -> anyhow::Result<PathBuf> {
    // First, check current project's .leann directory
    let local_path = PathBuf::from(".leann").join("indexes").join(name);
    if local_path.exists() {
        return Ok(local_path);
    }

    // Check if it's an absolute path
    let abs_path = PathBuf::from(name);
    if abs_path.is_absolute() && abs_path.exists() {
        return Ok(abs_path);
    }

    // Check home directory global registry
    if let Some(home) = dirs::home_dir() {
        let global_path = home.join(".leann").join("indexes").join(name);
        if global_path.exists() {
            return Ok(global_path);
        }
    }

    anyhow::bail!(
        "Index '{}' not found. Run 'leann list' to see available indexes.",
        name
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_index_not_found() {
        let result = find_index("nonexistent-index-12345");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
    }
}
