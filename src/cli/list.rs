//! List command - show all indexes

use std::path::PathBuf;

use clap::Args;

use crate::index::IndexMeta;

#[derive(Args)]
pub struct ListArgs {
    /// Show detailed information
    #[arg(short, long)]
    pub detailed: bool,
}

pub async fn run(args: ListArgs) -> anyhow::Result<()> {
    println!("LEANN Indexes");
    println!("{}", "=".repeat(50));

    let mut total_indexes = 0;

    // Check current project
    println!("\nCurrent Project");
    println!("   {}", std::env::current_dir()?.display());
    println!("   {}", "-".repeat(45));

    let local_path = PathBuf::from(".leann").join("indexes");
    if local_path.exists() {
        for entry in std::fs::read_dir(&local_path)? {
            let entry = entry?;
            if entry.path().is_dir() {
                let index_name = entry.file_name().to_string_lossy().to_string();
                let meta_path = entry.path().join("documents.leann.meta.json");

                let status = if meta_path.exists() { "OK" } else { "INCOMPLETE" };

                print!("   {}. {} {}", total_indexes + 1, index_name, status);

                if args.detailed && meta_path.exists() {
                    if let Ok(meta) = IndexMeta::load(&meta_path) {
                        print!(" ({} passages, {} dims)", meta.passage_count, meta.dimensions);
                    }
                }

                // Calculate size
                if let Ok(size) = calculate_dir_size(&entry.path()) {
                    print!(" [{:.1} MB]", size as f64 / (1024.0 * 1024.0));
                }

                println!();
                total_indexes += 1;
            }
        }

        if total_indexes == 0 {
            println!("   No indexes found");
        }
    } else {
        println!("   No .leann directory");
    }

    // Check home directory
    if let Some(home) = dirs::home_dir() {
        let global_path = home.join(".leann").join("indexes");
        if global_path.exists() {
            println!("\nGlobal Indexes (~/.leann/indexes)");
            println!("   {}", "-".repeat(45));

            let mut global_count = 0;
            for entry in std::fs::read_dir(&global_path)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    let index_name = entry.file_name().to_string_lossy().to_string();
                    let meta_path = entry.path().join("documents.leann.meta.json");

                    let status = if meta_path.exists() { "OK" } else { "INCOMPLETE" };

                    print!("   {}. {} {}", total_indexes + 1, index_name, status);

                    if args.detailed && meta_path.exists() {
                        if let Ok(meta) = IndexMeta::load(&meta_path) {
                            print!(" ({} passages, {} dims)", meta.passage_count, meta.dimensions);
                        }
                    }

                    if let Ok(size) = calculate_dir_size(&entry.path()) {
                        print!(" [{:.1} MB]", size as f64 / (1024.0 * 1024.0));
                    }

                    println!();
                    total_indexes += 1;
                    global_count += 1;
                }
            }

            if global_count == 0 {
                println!("   No indexes found");
            }
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("Total: {} index(es)", total_indexes);

    if total_indexes == 0 {
        println!("\nGet started:");
        println!("   leann build my-docs --docs ./documents");
    }

    Ok(())
}

fn calculate_dir_size(path: &PathBuf) -> std::io::Result<u64> {
    let mut size = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            size += metadata.len();
        } else if metadata.is_dir() {
            size += calculate_dir_size(&entry.path())?;
        }
    }
    Ok(size)
}
