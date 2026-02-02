//! Index searcher - query vector indexes

use std::path::Path;

use tracing::info;

use crate::backend::{BackendSearcher, BackendType};

use super::bm25::{Bm25Scorer, hybrid_rerank};
use super::filter::MetadataFilter;
use super::meta::IndexMeta;
use super::passages::PassageStore;

/// Search result with passage text and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub text: String,
    pub metadata: serde_json::Value,
}

/// Search options for advanced queries
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Number of results to return
    pub top_k: usize,
    /// Search complexity (HNSW expansion)
    pub complexity: usize,
    /// Metadata filter
    pub filter: Option<MetadataFilter>,
    /// Enable hybrid search (vector + BM25)
    pub hybrid: bool,
    /// Weight for vector scores in hybrid mode (0.0-1.0)
    pub hybrid_alpha: f32,
    /// Query text (for BM25 in hybrid mode)
    pub query_text: Option<String>,
}

impl SearchOptions {
    pub fn new(top_k: usize, complexity: usize) -> Self {
        Self {
            top_k,
            complexity,
            filter: None,
            hybrid: false,
            hybrid_alpha: 0.7,
            query_text: None,
        }
    }

    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_hybrid(mut self, query_text: String, alpha: f32) -> Self {
        self.hybrid = true;
        self.hybrid_alpha = alpha;
        self.query_text = Some(query_text);
        self
    }
}

/// Searcher for querying a LEANN index
pub struct IndexSearcher {
    passages: PassageStore,
    backend: Box<dyn BackendSearcher>,
    id_map: Vec<String>,
    /// All passage texts for BM25 (lazy-loaded)
    all_texts: Option<Vec<String>>,
}

impl IndexSearcher {
    /// Load an index for searching
    pub fn load(index_path: &Path, meta: &IndexMeta) -> anyhow::Result<Self> {
        info!("Loading index from {:?}", index_path);

        // Load passages
        let passages = PassageStore::open(index_path)?;

        // Load ID mapping
        let ids_path = index_path.with_extension("ids.txt");
        let id_map: Vec<String> = if ids_path.exists() {
            std::fs::read_to_string(&ids_path)?
                .lines()
                .map(|s| s.to_string())
                .collect()
        } else {
            // Fall back to passage store IDs
            passages.ids().cloned().collect()
        };

        // Load backend
        let backend_type = match meta.backend_name.as_str() {
            "hnsw" => BackendType::Hnsw,
            "diskann" => BackendType::DiskAnn,
            _ => anyhow::bail!("Unknown backend: {}", meta.backend_name),
        };

        let backend = backend_type.load_searcher(index_path, meta.dimensions)?;

        Ok(Self {
            passages,
            backend,
            id_map,
            all_texts: None,
        })
    }

    /// Simple search for nearest neighbors
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        complexity: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let opts = SearchOptions::new(top_k, complexity);
        self.search_with_options(query_embedding, &opts)
    }

    /// Advanced search with options
    pub fn search_with_options(
        &self,
        query_embedding: &[f32],
        opts: &SearchOptions,
    ) -> anyhow::Result<Vec<SearchResult>> {
        // Fetch more results if filtering or hybrid, to ensure we have enough after processing
        let fetch_k = if opts.filter.is_some() || opts.hybrid {
            opts.top_k * 5 // More for hybrid to get diverse results
        } else {
            opts.top_k
        };

        // Search backend
        let (indices, distances) = self.backend.search(query_embedding, fetch_k, opts.complexity)?;

        // Convert to (idx, score) pairs
        let mut vector_results: Vec<(usize, f32)> = indices
            .iter()
            .zip(distances.iter())
            .map(|(idx, dist)| (*idx as usize, *dist))
            .collect();

        // Apply hybrid search if enabled
        if opts.hybrid {
            if let Some(query_text) = &opts.query_text {
                // Load all texts for BM25 if not cached
                let all_texts = self.get_all_texts()?;
                let scorer = Bm25Scorer::build(&all_texts);
                let bm25_scores = scorer.score_query(query_text);

                // Get top BM25 results that might not be in vector results
                let bm25_top = scorer.search(query_text, fetch_k);

                // Add BM25 top results to vector results if not already present
                let vector_indices: std::collections::HashSet<usize> =
                    vector_results.iter().map(|(idx, _)| *idx).collect();

                for (idx, _bm25_score) in bm25_top {
                    if !vector_indices.contains(&idx) {
                        // Add with a low vector score (will be boosted by BM25)
                        vector_results.push((idx, 0.0));
                    }
                }

                vector_results = hybrid_rerank(&vector_results, &bm25_scores, opts.hybrid_alpha);
            }
        }

        // Convert to SearchResults and apply filtering
        let mut results = Vec::with_capacity(opts.top_k);

        for (idx, score) in vector_results {
            if results.len() >= opts.top_k {
                break;
            }

            // Map integer index to string ID
            let id = if idx < self.id_map.len() {
                self.id_map[idx].clone()
            } else {
                idx.to_string()
            };

            // Get passage text and metadata
            match self.passages.get(&id) {
                Ok(passage) => {
                    // Apply metadata filter
                    if let Some(filter) = &opts.filter {
                        if !filter.matches(&passage.metadata) {
                            continue;
                        }
                    }

                    results.push(SearchResult {
                        id,
                        score,
                        text: passage.text,
                        metadata: passage.metadata,
                    });
                }
                Err(e) => {
                    tracing::warn!("Failed to load passage {}: {}", id, e);
                }
            }
        }

        Ok(results)
    }

    /// Get all passage texts for BM25
    fn get_all_texts(&self) -> anyhow::Result<Vec<String>> {
        let mut texts = Vec::with_capacity(self.id_map.len());

        for id in &self.id_map {
            match self.passages.get(id) {
                Ok(passage) => texts.push(passage.text),
                Err(_) => texts.push(String::new()),
            }
        }

        Ok(texts)
    }

    /// BM25-only search for query expansion
    /// Returns passage texts of top matches
    pub fn bm25_search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<String>> {
        let all_texts = self.get_all_texts()?;
        let scorer = Bm25Scorer::build(&all_texts);
        let results = scorer.search(query, top_k);

        let texts: Vec<String> = results
            .iter()
            .filter_map(|(idx, _)| {
                if *idx < self.id_map.len() {
                    let id = &self.id_map[*idx];
                    self.passages.get(id).ok().map(|p| p.text)
                } else {
                    None
                }
            })
            .collect();

        Ok(texts)
    }

    /// Get passage count
    pub fn len(&self) -> usize {
        self.backend.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.backend.is_empty()
    }
}
