//! Remove command - delete an index

use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct RemoveArgs {
    /// Index name to remove
    pub index_name: String,

    /// Force removal without confirmation
    #[arg(short, long)]
    pub force: bool,
}

pub async fn run(args: RemoveArgs) -> anyhow::Result<()> {
    // Find all matching indexes
    let matches = find_all_indexes(&args.index_name)?;

    if matches.is_empty() {
        anyhow::bail!("Index '{}' not found.", args.index_name);
    }

    if matches.len() == 1 {
        let (path, location) = &matches[0];
        remove_index(path, &args.index_name, location, args.force)?;
    } else {
        println!("Found {} indexes named '{}':", matches.len(), args.index_name);
        for (i, (path, location)) in matches.iter().enumerate() {
            println!("   {}. {} ({})", i + 1, path.display(), location);
        }

        if args.force {
            anyhow::bail!("Multiple matches found. Cannot use --force with multiple indexes.");
        }

        print!("Which one to remove? (1-{}, or 'c' to cancel): ", matches.len());
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "c" || input == "cancel" {
            println!("Cancelled.");
            return Ok(());
        }

        let choice: usize = input.parse().map_err(|_| anyhow::anyhow!("Invalid choice"))?;
        if choice < 1 || choice > matches.len() {
            anyhow::bail!("Invalid choice");
        }

        let (path, location) = &matches[choice - 1];
        remove_index(path, &args.index_name, location, false)?;
    }

    Ok(())
}

fn find_all_indexes(name: &str) -> anyhow::Result<Vec<(PathBuf, String)>> {
    let mut matches = Vec::new();

    // Check current project
    let local_path = PathBuf::from(".leann").join("indexes").join(name);
    if local_path.exists() {
        matches.push((local_path, "current project".to_string()));
    }

    // Check home directory
    if let Some(home) = dirs::home_dir() {
        let global_path = home.join(".leann").join("indexes").join(name);
        if global_path.exists() {
            matches.push((global_path, "global (~/.leann)".to_string()));
        }
    }

    Ok(matches)
}

fn remove_index(
    path: &PathBuf,
    name: &str,
    location: &str,
    force: bool,
) -> anyhow::Result<()> {
    if !force {
        print!(
            "Remove index '{}' from {}? Type '{}' to confirm: ",
            name, location, name
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim() != name {
            println!("Confirmation failed. Index not removed.");
            return Ok(());
        }
    }

    std::fs::remove_dir_all(path)?;
    println!("Index '{}' removed from {}.", name, location);

    Ok(())
}
