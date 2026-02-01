//! Prune command - delete embeddings to enable recomputation mode

use std::path::PathBuf;

use clap::Args;

use crate::index::{prune_embeddings, EmbeddingsStore, IndexMeta};

#[derive(Args)]
pub struct PruneArgs {
    /// Index name to prune
    pub index_name: String,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,
}

pub async fn run(args: PruneArgs) -> anyhow::Result<()> {
    // Find index
    let index_dir = find_index(&args.index_name)?;
    let meta_path = index_dir.join("documents.leann.meta.json");
    let index_path = index_dir.join("documents.leann");

    // Load metadata
    let mut meta = IndexMeta::load(&meta_path)?;

    // Check if embeddings exist
    let embeddings_path = EmbeddingsStore::path_for_index(&index_path);
    if !embeddings_path.exists() {
        if meta.is_pruned {
            println!("Index '{}' is already pruned (no embeddings file)", args.index_name);
        } else {
            println!(
                "Index '{}' has no separate embeddings file. \
                Rebuild with --recompute flag to enable pruning.",
                args.index_name
            );
        }
        return Ok(());
    }

    // Get file size
    let file_size = std::fs::metadata(&embeddings_path)?.len();
    let size_mb = file_size as f64 / (1024.0 * 1024.0);

    // Confirm
    if !args.yes {
        println!(
            "This will delete the embeddings file for index '{}' ({:.2} MB).",
            args.index_name, size_mb
        );
        println!("Embeddings will be recomputed on-demand during search.");
        println!();
        print!("Type '{}' to confirm: ", args.index_name);
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim() != args.index_name {
            println!("Confirmation failed. Index not pruned.");
            return Ok(());
        }
    }

    // Delete embeddings
    prune_embeddings(&index_path)?;

    // Update metadata
    meta.is_pruned = true;
    meta.save(&meta_path)?;

    println!(
        "Index '{}' pruned. Saved {:.2} MB of storage.",
        args.index_name, size_mb
    );
    println!("Note: Search will now recompute embeddings on-demand (slower).");

    Ok(())
}

/// Find an index by name
fn find_index(name: &str) -> anyhow::Result<PathBuf> {
    let local_path = PathBuf::from(".leann").join("indexes").join(name);
    if local_path.exists() {
        return Ok(local_path);
    }

    let abs_path = PathBuf::from(name);
    if abs_path.is_absolute() && abs_path.exists() {
        return Ok(abs_path);
    }

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
