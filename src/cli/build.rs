//! Build command - index construction from documents

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use crate::backend::BackendType;
use crate::chunker::{Chunk, Chunker, ChunkingStrategy, SmartChunker};
use crate::config::Config;
use crate::embedding::{get_model_config, EmbeddingMode, EmbeddingProvider};
use crate::index::{IndexMeta, StreamingIndexBuilder};

#[derive(Args)]
pub struct BuildArgs {
    /// Index name (default: current directory name)
    #[arg()]
    pub index_name: Option<String>,

    /// Document directories and/or files
    #[arg(long, default_value = ".")]
    pub docs: Vec<PathBuf>,

    /// Backend to use
    #[arg(long, default_value = "hnsw", value_parser = ["hnsw", "diskann"])]
    pub backend_name: String,

    /// Embedding model name
    #[arg(long, default_value = "text-embedding-3-small")]
    pub embedding_model: String,

    /// Embedding mode
    #[cfg(feature = "local-embeddings")]
    #[arg(long, default_value = "openai", value_parser = ["openai", "ollama", "gemini", "local"])]
    pub embedding_mode: String,

    /// Embedding mode
    #[cfg(not(feature = "local-embeddings"))]
    #[arg(long, default_value = "openai", value_parser = ["openai", "ollama", "gemini"])]
    pub embedding_mode: String,

    /// Ollama host for embeddings
    #[arg(long, env = "OLLAMA_HOST")]
    pub embedding_host: Option<String>,

    /// OpenAI API base URL
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub embedding_api_base: Option<String>,

    /// API key for embedding service (OpenAI)
    #[arg(long, env = "OPENAI_API_KEY")]
    pub embedding_api_key: Option<String>,

    /// Google API key for Gemini embeddings
    #[arg(long, env = "GOOGLE_API_KEY")]
    pub google_api_key: Option<String>,

    /// Prompt template prefix for document embeddings
    /// Used for asymmetric models (e.g., "passage: " for E5, BGE models)
    #[arg(long)]
    pub embedding_prompt_template: Option<String>,

    /// Local model path (for local embedding mode)
    #[cfg(feature = "local-embeddings")]
    #[arg(long)]
    pub embedding_model_path: Option<String>,

    /// Force rebuild existing index
    #[arg(short, long)]
    pub force: bool,

    /// Enable recomputation mode (stores embeddings separately for pruning)
    #[arg(long)]
    pub recompute: bool,

    /// Graph degree for HNSW
    #[arg(long, default_value = "32")]
    pub graph_degree: usize,

    /// Build complexity
    #[arg(long, default_value = "64")]
    pub complexity: usize,

    /// Document chunk size in tokens
    #[arg(long, default_value = "256")]
    pub doc_chunk_size: usize,

    /// Document chunk overlap in tokens
    #[arg(long, default_value = "128")]
    pub doc_chunk_overlap: usize,

    /// File types to include (comma-separated, e.g., ".txt,.pdf,.md")
    #[arg(long)]
    pub file_types: Option<String>,

    /// File types to exclude (comma-separated, e.g., ".json,.lock")
    /// Default excludes: .json (often large data files with poor semantic value)
    #[arg(long)]
    pub exclude_types: Option<String>,

    /// Include hidden files
    #[arg(long)]
    pub include_hidden: bool,

    /// Chunking strategy: simple, ast, or auto (default: simple)
    /// - simple: character-based chunking with word boundary awareness (recommended)
    /// - ast: AST-aware chunking for code files (functions, classes, etc.) - experimental
    /// - auto: automatically selects based on file type - experimental
    #[arg(long, default_value = "simple", value_parser = ["simple", "ast", "auto"])]
    pub chunking_strategy: String,

    /// Batch size for embedding API calls (default: provider-specific)
    #[arg(long)]
    pub embedding_batch_size: Option<usize>,

    /// Maximum number of files to index (for testing on large repos)
    #[arg(long)]
    pub max_files: Option<usize>,

    /// Skip files larger than this size in KB (default: 1024 = 1MB)
    #[arg(long, default_value = "1024")]
    pub max_file_size_kb: usize,
}

pub async fn run(args: BuildArgs, _verbose: bool) -> anyhow::Result<()> {
    // Load config file for defaults
    let config = Config::load();

    let index_name = args.index_name.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "index".to_string())
    });

    info!("Building index '{}'", index_name);

    // Determine index directory
    let index_dir = PathBuf::from(".leann").join("indexes").join(&index_name);

    if index_dir.exists() && !args.force {
        anyhow::bail!(
            "Index '{}' already exists. Use --force to rebuild.",
            index_name
        );
    }

    // Create index directory
    std::fs::create_dir_all(&index_dir)?;

    // Parse backend type
    let backend_type = match args.backend_name.as_str() {
        "hnsw" => BackendType::Hnsw,
        "diskann" => BackendType::DiskAnn,
        _ => anyhow::bail!("Unknown backend: {}", args.backend_name),
    };

    // Use config defaults if CLI args not provided
    let embedding_mode_str = if args.embedding_mode == "openai" && config.embedding.provider != "openai" {
        // CLI default is "openai", check if config overrides it
        config.embedding.provider.as_str()
    } else {
        args.embedding_mode.as_str()
    };

    // Use config model if CLI uses default
    let embedding_model = if args.embedding_model == "text-embedding-3-small" {
        config.embedding.model.clone()
    } else {
        args.embedding_model.clone()
    };

    // Parse embedding mode, using config for host/base_url if not specified in CLI
    let embedding_mode = match embedding_mode_str {
        "openai" | "lmstudio" => EmbeddingMode::OpenAI {
            api_key: args.embedding_api_key.clone().or(config.embedding.api_key.clone()),
            base_url: args.embedding_api_base.clone().or(config.embedding.base_url.clone()),
        },
        "ollama" => EmbeddingMode::Ollama {
            host: args.embedding_host.clone().or(config.embedding.host.clone()),
        },
        "gemini" => EmbeddingMode::Gemini {
            api_key: args.google_api_key.clone().or(config.embedding.api_key.clone()),
        },
        #[cfg(feature = "local-embeddings")]
        "local" => EmbeddingMode::Local {
            model_path: args.embedding_model_path.clone(),
        },
        _ => anyhow::bail!("Unknown embedding mode: {}", embedding_mode_str),
    };

    // Get model-specific configuration (prefixes, normalization)
    let model_config = get_model_config(&embedding_model);
    let document_prefix = args.embedding_prompt_template
        .clone()
        .unwrap_or_else(|| model_config.document_prefix.to_string());

    info!("Using embedding: {} / {}", embedding_mode_str, embedding_model);
    if !document_prefix.is_empty() {
        info!("Document prefix: {:?}", document_prefix);
    }

    // Create embedding provider
    let embedding_provider = EmbeddingProvider::new(
        embedding_model.clone(),
        embedding_mode.clone(),
    ).await?;

    // Get embedding dimensions
    let dimensions = embedding_provider.dimensions();
    info!("Embedding dimensions: {}", dimensions);

    // Load documents
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    progress.set_message("Loading documents...");

    let file_types: Option<Vec<String>> = args.file_types.map(|ft| {
        ft.split(',')
            .map(|s| s.trim().to_string())
            .collect()
    });

    let chunking_strategy: ChunkingStrategy = args.chunking_strategy.parse()
        .unwrap_or(ChunkingStrategy::Auto);

    // Collect file paths first (low memory), then process in batches
    let exclude_types: Option<Vec<String>> = args.exclude_types.map(|ft| {
        ft.split(',')
            .map(|s| s.trim().to_string())
            .collect()
    });

    let file_paths = collect_file_paths(
        &args.docs,
        file_types.as_deref(),
        exclude_types.as_deref(),
        args.include_hidden,
        args.max_files,
        args.max_file_size_kb,
    )?;

    progress.finish_with_message(format!("Found {} files", file_paths.len()));

    if file_paths.is_empty() {
        anyhow::bail!("No documents found to index");
    }


    // Build index using streaming builder to minimize memory usage
    let index_path = index_dir.join("documents.leann");
    let mut builder = StreamingIndexBuilder::new(
        backend_type,
        dimensions,
        args.graph_degree,
        args.complexity,
        args.recompute,
        &index_path,
    )?;


    // Process files in streaming fashion to avoid memory explosion
    let batch_size = args.embedding_batch_size.unwrap_or_else(|| {
        match embedding_mode_str {
            "ollama" => 32,
            _ => 100,
        }
    });

    let chunker = SmartChunker::new(chunking_strategy, args.doc_chunk_size, args.doc_chunk_overlap);

    let progress = ProgressBar::new(file_paths.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut total_chunks = 0usize;
    let mut chunk_id = 0u64;
    let mut pending_chunks: Vec<Chunk> = Vec::with_capacity(batch_size);

    // Statistics for diagnostics
    let mut stats: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new(); // ext -> (files, chunks)
    let mut embed_time_total = std::time::Duration::ZERO;
    let mut embed_batches = 0usize;
    let build_start = std::time::Instant::now();

    for file_path in &file_paths {
        // Load and chunk one file at a time
        if let Some(content) = load_file_content(file_path) {
            let file_chunks = chunker.chunk(&content, file_path, &mut chunk_id);

            // Track stats by extension
            let ext = file_path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_string();
            let entry = stats.entry(ext).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += file_chunks.len();

            pending_chunks.extend(file_chunks);

            // Process batch when full
            while pending_chunks.len() >= batch_size {
                let batch: Vec<Chunk> = pending_chunks.drain(..batch_size).collect();
                let batch_start = std::time::Instant::now();
                process_chunk_batch(&batch, &embedding_provider, &document_prefix, &mut builder).await?;
                embed_time_total += batch_start.elapsed();
                embed_batches += 1;
                total_chunks += batch.len();
                // Log progress every 500 chunks
                if total_chunks % 500 == 0 {
                    let elapsed = build_start.elapsed().as_secs();
                    let rate = if elapsed > 0 { total_chunks / elapsed as usize } else { 0 };
                    info!("Progress: {} chunks, {} chunks/sec, avg batch: {:.0}ms",
                          total_chunks, rate,
                          embed_time_total.as_millis() as f64 / embed_batches as f64);
                }
            }
        }
        progress.inc(1);
    }

    // Process remaining chunks
    if !pending_chunks.is_empty() {
        process_chunk_batch(&pending_chunks, &embedding_provider, &document_prefix, &mut builder).await?;
        total_chunks += pending_chunks.len();
    }

    progress.finish_with_message(format!("Indexed {} chunks from {} files", total_chunks, file_paths.len()));

    // Build the vector index
    builder.build()?;

    // Save metadata with embedding options (including query prefix for search)
    let query_prefix = model_config.query_prefix.to_string();
    let embedding_options = if !query_prefix.is_empty() || !document_prefix.is_empty() {
        Some(serde_json::json!({
            "query_prompt_template": query_prefix,
            "build_prompt_template": document_prefix,
        }))
    } else {
        None
    };

    let meta = IndexMeta {
        version: "1.0".to_string(),
        backend_name: args.backend_name,
        embedding_model,
        embedding_mode: embedding_mode_str.to_string(),
        dimensions,
        passage_count: total_chunks,
        backend_kwargs: None,
        embedding_options,
        is_recompute: args.recompute,
        is_pruned: false,
    };
    meta.save(&index_dir.join("documents.leann.meta.json"))?;

    let total_time = build_start.elapsed();
    println!("Index '{}' built successfully at {:?}", index_name, index_dir);
    println!("  Passages: {}", total_chunks);
    println!("  Dimensions: {}", dimensions);
    println!("  Total time: {:.1}s", total_time.as_secs_f64());
    println!("  Embedding time: {:.1}s ({:.0}%)",
             embed_time_total.as_secs_f64(),
             100.0 * embed_time_total.as_secs_f64() / total_time.as_secs_f64());
    println!("  Avg batch time: {:.0}ms ({} batches)",
             embed_time_total.as_millis() as f64 / embed_batches.max(1) as f64,
             embed_batches);

    // Show top file types by chunk count
    let mut stats_vec: Vec<_> = stats.into_iter().collect();
    stats_vec.sort_by(|a, b| b.1.1.cmp(&a.1.1)); // Sort by chunk count desc
    println!("\n  Top file types by chunks:");
    for (ext, (files, chunks)) in stats_vec.iter().take(10) {
        let avg = if *files > 0 { *chunks / *files } else { 0 };
        println!("    .{}: {} files, {} chunks (avg {}/file)", ext, files, chunks, avg);
    }

    if args.recompute {
        println!("\n  Recompute mode: enabled (run 'leann prune {}' to save space)", index_name);
    }

    Ok(())
}

/// Process a batch of chunks: compute embeddings and add to builder
async fn process_chunk_batch(
    chunks: &[Chunk],
    embedding_provider: &EmbeddingProvider,
    embed_template: &str,
    builder: &mut StreamingIndexBuilder,
) -> anyhow::Result<()> {
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = embedding_provider.embed_with_template(&texts, embed_template).await?;

    for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
        builder.add_passage(&chunk.id, &chunk.text, embedding, chunk.metadata.clone())?;
    }

    Ok(())
}

/// Collect file paths without loading content (memory efficient)
fn collect_file_paths(
    paths: &[PathBuf],
    file_types: Option<&[String]>,
    exclude_types: Option<&[String]>,
    include_hidden: bool,
    max_files: Option<usize>,
    max_file_size_kb: usize,
) -> anyhow::Result<Vec<PathBuf>> {
    use ignore::WalkBuilder;

    let max_file_bytes = max_file_size_kb as u64 * 1024;
    let mut file_paths = Vec::new();

    // Default file types
    #[cfg(feature = "pdf")]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
        ".pdf",
    ];
    #[cfg(not(feature = "pdf"))]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
    ];

    // File types with stricter size limits (JSON often contains data, not semantic content)
    let strict_size_extensions: std::collections::HashSet<&str> = [".json"].into_iter().collect();
    let strict_size_limit: u64 = 10 * 1024; // 10KB for JSON files

    let allowed_extensions: Vec<&str> = file_types
        .map(|ft| ft.iter().map(|s| s.as_str()).collect())
        .unwrap_or(default_types);

    // Extensions to exclude (user can override with --exclude-types)
    let excluded_extensions: Vec<&str> = exclude_types
        .map(|ft| ft.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default();

    for path in paths {
        if let Some(max) = max_files {
            if file_paths.len() >= max {
                tracing::info!("Reached max files limit ({})", max);
                break;
            }
        }

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = format!(".{}", ext.to_string_lossy());

                // Skip excluded extensions
                if excluded_extensions.iter().any(|e| *e == ext_str) {
                    continue;
                }
                // Skip if not in allowed list
                if !allowed_extensions.iter().any(|e| *e == ext_str) {
                    continue;
                }

                // Check file size (stricter limit for JSON-like files)
                if let Ok(metadata) = path.metadata() {
                    let size_limit = if strict_size_extensions.contains(ext_str.as_str()) {
                        strict_size_limit
                    } else {
                        max_file_bytes
                    };
                    if metadata.len() > size_limit {
                        tracing::debug!("Skipping large file: {} ({}KB, limit {}KB)",
                            path.display(), metadata.len() / 1024, size_limit / 1024);
                        continue;
                    }
                }
                file_paths.push(path.clone());
            }
        } else if path.is_dir() {
            let walker = WalkBuilder::new(path)
                .hidden(!include_hidden)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                // Add common build/dependency directories to ignore
                .add_custom_ignore_filename(".leannignore")
                .filter_entry(|entry| {
                    let name = entry.file_name().to_string_lossy();
                    // Skip common build/dependency directories
                    !matches!(name.as_ref(),
                        "target" | "node_modules" | ".git" | "__pycache__" |
                        "venv" | ".venv" | "dist" | "build" | ".next" |
                        ".nuxt" | "vendor" | "Pods" | ".gradle" | ".cache" |
                        "deps" | "_build" | ".elixir_ls" | ".hex" | "priv"
                    )
                })
                .build();

            for entry in walker.flatten() {
                if let Some(max) = max_files {
                    if file_paths.len() >= max {
                        tracing::info!("Reached max files limit ({})", max);
                        break;
                    }
                }

                let entry_path = entry.path();
                if entry_path.is_file() {
                    if let Some(ext) = entry_path.extension() {
                        let ext_str = format!(".{}", ext.to_string_lossy());

                        // Skip excluded extensions
                        if excluded_extensions.iter().any(|e| *e == ext_str) {
                            continue;
                        }
                        // Skip if not in allowed list
                        if !allowed_extensions.iter().any(|e| *e == ext_str) {
                            continue;
                        }

                        // Check file size (stricter limit for JSON-like files)
                        if let Ok(metadata) = entry_path.metadata() {
                            let size_limit = if strict_size_extensions.contains(ext_str.as_str()) {
                                strict_size_limit
                            } else {
                                max_file_bytes
                            };
                            if metadata.len() > size_limit {
                                tracing::debug!("Skipping large file: {} ({}KB, limit {}KB)",
                                    entry_path.display(), metadata.len() / 1024, size_limit / 1024);
                                continue;
                            }
                        }
                        file_paths.push(entry_path.to_path_buf());
                    }
                }
            }
        }
    }

    Ok(file_paths)
}

/// Load documents from paths and chunk them
pub fn load_documents(
    paths: &[PathBuf],
    chunk_size: usize,
    chunk_overlap: usize,
    file_types: Option<&[String]>,
    include_hidden: bool,
    chunking_strategy: ChunkingStrategy,
) -> anyhow::Result<Vec<Chunk>> {
    use ignore::WalkBuilder;

    let chunker = SmartChunker::new(chunking_strategy, chunk_size, chunk_overlap);
    let mut chunks = Vec::new();
    let mut chunk_id = 0u64;

    // Default file types
    #[cfg(feature = "pdf")]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
        ".pdf",
    ];
    #[cfg(not(feature = "pdf"))]
    let default_types = vec![
        ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
    ];

    let allowed_extensions: Vec<&str> = file_types
        .map(|ft| ft.iter().map(|s| s.as_str()).collect())
        .unwrap_or(default_types);

    for path in paths {
        if path.is_file() {
            // Single file
            if let Some(ext) = path.extension() {
                let ext_str = format!(".{}", ext.to_string_lossy());
                if allowed_extensions.iter().any(|e| *e == ext_str) {
                    if let Some(content) = load_file_content(path) {
                        let file_chunks = chunker.chunk(&content, path, &mut chunk_id);
                        chunks.extend(file_chunks);
                    }
                }
            }
        } else if path.is_dir() {
            // Directory - walk with gitignore support
            let walker = WalkBuilder::new(path)
                .hidden(!include_hidden)
                .git_ignore(true)
                .git_global(true)
                .build();

            for entry in walker.flatten() {
                let entry_path = entry.path();
                if entry_path.is_file() {
                    if let Some(ext) = entry_path.extension() {
                        let ext_str = format!(".{}", ext.to_string_lossy());
                        if allowed_extensions.iter().any(|e| *e == ext_str) {
                            if let Some(content) = load_file_content(entry_path) {
                                let file_chunks = chunker.chunk(&content, entry_path, &mut chunk_id);
                                chunks.extend(file_chunks);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(chunks)
}

/// Load file content, handling different file types
fn load_file_content(path: &std::path::Path) -> Option<String> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        #[cfg(feature = "pdf")]
        "pdf" => {
            match pdf_extract::extract_text(path) {
                Ok(text) => {
                    let text = text.trim();
                    if text.is_empty() {
                        tracing::warn!("PDF {} contains no extractable text", path.display());
                        None
                    } else {
                        Some(text.to_string())
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to extract text from {}: {}", path.display(), e);
                    None
                }
            }
        }
        _ => {
            // Regular text file
            std::fs::read_to_string(path).ok()
        }
    }
}
