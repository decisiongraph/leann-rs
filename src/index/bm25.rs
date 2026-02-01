//! BM25 scoring for hybrid search

use std::sync::LazyLock;

use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};

/// BM25 parameters
const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Cached regex for tokenization (compiled once)
static TOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-zA-Z0-9]+").unwrap()
});

/// Simple BM25 scorer
pub struct Bm25Scorer {
    /// Document frequency: term -> number of documents containing term
    doc_freq: FxHashMap<String, usize>,
    /// Total number of documents
    num_docs: usize,
    /// Average document length
    avg_doc_len: f32,
    /// Document lengths
    doc_lengths: Vec<usize>,
    /// Term frequencies per document: doc_id -> (term -> count)
    term_freqs: Vec<FxHashMap<String, usize>>,
}

impl Bm25Scorer {
    /// Build a BM25 scorer from documents
    pub fn build(documents: &[String]) -> Self {
        let num_docs = documents.len();
        let mut doc_freq: FxHashMap<String, usize> = FxHashMap::default();
        let mut doc_lengths = Vec::with_capacity(num_docs);
        let mut term_freqs = Vec::with_capacity(num_docs);
        let mut total_len = 0usize;

        for doc in documents {
            let tokens = tokenize(doc);
            let doc_len = tokens.len();
            doc_lengths.push(doc_len);
            total_len += doc_len;

            let mut tf: FxHashMap<String, usize> = FxHashMap::default();
            let mut seen: FxHashSet<String> = FxHashSet::default();

            for token in tokens {
                *tf.entry(token.clone()).or_insert(0) += 1;

                if !seen.contains(&token) {
                    *doc_freq.entry(token.clone()).or_insert(0) += 1;
                    seen.insert(token);
                }
            }

            term_freqs.push(tf);
        }

        let avg_doc_len = if num_docs > 0 {
            total_len as f32 / num_docs as f32
        } else {
            1.0
        };

        Self {
            doc_freq,
            num_docs,
            avg_doc_len,
            doc_lengths,
            term_freqs,
        }
    }

    /// Score a query against all documents
    pub fn score_query(&self, query: &str) -> Vec<f32> {
        let query_tokens = tokenize(query);
        let mut scores = vec![0.0f32; self.num_docs];

        for token in &query_tokens {
            let df = *self.doc_freq.get(token).unwrap_or(&0) as f32;
            if df == 0.0 {
                continue;
            }

            // IDF component
            let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            for (doc_id, tf_map) in self.term_freqs.iter().enumerate() {
                let tf = *tf_map.get(token).unwrap_or(&0) as f32;
                if tf == 0.0 {
                    continue;
                }

                let doc_len = self.doc_lengths[doc_id] as f32;
                let norm = 1.0 - B + B * (doc_len / self.avg_doc_len);

                // BM25 score component
                let score = idf * (tf * (K1 + 1.0)) / (tf + K1 * norm);
                scores[doc_id] += score;
            }
        }

        scores
    }

    /// Get top-k documents by BM25 score
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f32)> {
        let scores = self.score_query(query);

        let mut scored: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
    }
}

/// Simple tokenization: lowercase, split on non-alphanumeric
/// Uses cached regex for performance
fn tokenize(text: &str) -> Vec<String> {
    TOKEN_REGEX.find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .filter(|s| s.len() > 1) // Skip single-character tokens
        .collect()
}

/// Combine vector scores with BM25 scores
pub fn hybrid_rerank(
    vector_results: &[(usize, f32)],
    bm25_scores: &[f32],
    alpha: f32, // Weight for vector scores (1-alpha for BM25)
) -> Vec<(usize, f32)> {
    // Normalize vector scores
    let max_vector = vector_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_vector = vector_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let vector_range = (max_vector - min_vector).max(1e-6);

    // Normalize BM25 scores
    let max_bm25 = bm25_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_bm25 = bm25_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let bm25_range = (max_bm25 - min_bm25).max(1e-6);

    let mut combined: Vec<(usize, f32)> = vector_results
        .iter()
        .map(|(idx, vec_score)| {
            let norm_vec = (vec_score - min_vector) / vector_range;
            let bm25 = bm25_scores.get(*idx).copied().unwrap_or(0.0);
            let norm_bm25 = (bm25 - min_bm25) / bm25_range;

            let combined_score = alpha * norm_vec + (1.0 - alpha) * norm_bm25;
            (*idx, combined_score)
        })
        .collect();

    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single chars should be filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("test123 456abc");
        assert!(tokens.contains(&"test123".to_string()));
        assert!(tokens.contains(&"456abc".to_string()));
    }

    #[test]
    fn test_bm25_basic_scoring() {
        let docs = vec![
            "the quick brown fox jumps over the lazy dog".to_string(),
            "a quick brown dog outpaces a swift fox".to_string(),
            "the dog chases the fox around the yard".to_string(),
        ];

        let scorer = Bm25Scorer::build(&docs);
        let results = scorer.search("quick fox", 3);

        assert!(!results.is_empty());
        // First two docs should score higher (both have "quick" and "fox")
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_bm25_term_frequency_matters() {
        let docs = vec![
            "rust rust rust programming".to_string(),  // 3x "rust"
            "rust programming".to_string(),            // 1x "rust"
        ];

        let scorer = Bm25Scorer::build(&docs);
        let scores = scorer.score_query("rust");

        // Doc with more "rust" should score higher
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_bm25_idf_matters() {
        let docs = vec![
            "common rare".to_string(),
            "common".to_string(),
            "common".to_string(),
        ];

        let scorer = Bm25Scorer::build(&docs);
        let scores = scorer.score_query("rare");

        // Only first doc has "rare", so only it should score
        assert!(scores[0] > 0.0);
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn test_bm25_empty_query() {
        let docs = vec!["hello world".to_string()];
        let scorer = Bm25Scorer::build(&docs);
        let scores = scorer.score_query("");

        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn test_bm25_no_match() {
        let docs = vec!["hello world".to_string()];
        let scorer = Bm25Scorer::build(&docs);
        let results = scorer.search("xyz", 5);

        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_search_top_k() {
        let docs = vec![
            "apple banana".to_string(),
            "apple cherry".to_string(),
            "banana cherry".to_string(),
            "apple apple apple".to_string(),
        ];

        let scorer = Bm25Scorer::build(&docs);
        let results = scorer.search("apple", 2);

        // Should return only top 2
        assert_eq!(results.len(), 2);
        // Doc 3 (apple apple apple) should be first
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_hybrid_rerank_basic() {
        let vector_results = vec![
            (0, 0.9),
            (1, 0.8),
            (2, 0.7),
        ];
        let bm25_scores = vec![0.5, 0.9, 0.3];

        // Equal weight (alpha = 0.5)
        let results = hybrid_rerank(&vector_results, &bm25_scores, 0.5);

        assert_eq!(results.len(), 3);
        // All scores should be in [0, 1] range after normalization
        for (_, score) in &results {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_hybrid_rerank_vector_only() {
        let vector_results = vec![
            (0, 0.9),
            (1, 0.5),
        ];
        let bm25_scores = vec![0.1, 0.9];

        // alpha = 1.0 means vector only
        let results = hybrid_rerank(&vector_results, &bm25_scores, 1.0);

        // Doc 0 should be first (highest vector score)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hybrid_rerank_bm25_only() {
        let vector_results = vec![
            (0, 0.9),
            (1, 0.5),
        ];
        let bm25_scores = vec![0.1, 0.9];

        // alpha = 0.0 means BM25 only
        let results = hybrid_rerank(&vector_results, &bm25_scores, 0.0);

        // Doc 1 should be first (highest BM25 score)
        assert_eq!(results[0].0, 1);
    }
}
