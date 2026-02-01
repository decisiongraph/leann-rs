//! Benchmarks for LEANN core operations

use std::sync::LazyLock;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};

// Cached regex for optimized tokenization
static TOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-zA-Z0-9]+").unwrap()
});

/// Generate sample documents for benchmarking
fn generate_docs(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!(
            "This is document number {} with some content about programming, \
             Rust, Python, and machine learning. It contains keywords like \
             vector database, embedding, search, and retrieval. Document {}.",
            i, i
        ))
        .collect()
}

/// Benchmark dot product calculation (core of vector search)
fn bench_dot_product(c: &mut Criterion) {
    let dims = 1536; // OpenAI embedding size
    let a: Vec<f32> = (0..dims).map(|i| (i as f32) / 1000.0).collect();
    let b: Vec<f32> = (0..dims).map(|i| (i as f32) / 1000.0).collect();

    c.bench_function("dot_product_1536d", |bencher| {
        bencher.iter(|| {
            let sum: f32 = a.iter()
                .zip(b.iter())
                .map(|(x, y)| x * y)
                .sum();
            black_box(sum)
        });
    });

    // Also bench with SIMD-friendly dimensions
    let dims_768 = 768;
    let a768: Vec<f32> = (0..dims_768).map(|i| (i as f32) / 1000.0).collect();
    let b768: Vec<f32> = (0..dims_768).map(|i| (i as f32) / 1000.0).collect();

    c.bench_function("dot_product_768d", |bencher| {
        bencher.iter(|| {
            let sum: f32 = a768.iter()
                .zip(b768.iter())
                .map(|(x, y)| x * y)
                .sum();
            black_box(sum)
        });
    });
}

/// Benchmark tokenization (used in BM25) - uses cached regex
fn bench_tokenization(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. \
                Programming in Rust is fast and safe. \
                Machine learning models use vector embeddings.";

    c.bench_function("tokenize_sentence", |bencher| {
        bencher.iter(|| {
            let tokens: Vec<String> = TOKEN_REGEX
                .find_iter(black_box(text))
                .map(|m| m.as_str().to_lowercase())
                .filter(|s| s.len() > 1)
                .collect();
            black_box(tokens)
        });
    });

    // Longer text
    let long_text = text.repeat(100);
    c.bench_function("tokenize_long_text", |bencher| {
        bencher.iter(|| {
            let tokens: Vec<String> = TOKEN_REGEX
                .find_iter(black_box(&long_text))
                .map(|m| m.as_str().to_lowercase())
                .filter(|s| s.len() > 1)
                .collect();
            black_box(tokens)
        });
    });
}

/// Benchmark BM25 index building with optimized FxHashMap
fn bench_bm25_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_build_optimized");

    for size in [100, 1000, 10000].iter() {
        let docs = generate_docs(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Optimized BM25 build with FxHashMap and cached regex
                let mut doc_freq: FxHashMap<String, usize> = FxHashMap::default();

                for doc in &docs {
                    let mut seen: FxHashSet<String> = FxHashSet::default();
                    for m in TOKEN_REGEX.find_iter(doc) {
                        let token = m.as_str().to_lowercase();
                        if token.len() > 1 && !seen.contains(&token) {
                            *doc_freq.entry(token.clone()).or_insert(0) += 1;
                            seen.insert(token);
                        }
                    }
                }
                black_box(doc_freq)
            });
        });
    }
    group.finish();
}

/// Benchmark BM25 query scoring with optimized FxHashMap
fn bench_bm25_query(c: &mut Criterion) {
    // Pre-build a simple BM25 index with FxHashMap
    let docs = generate_docs(1000);

    // Build term frequencies
    let mut doc_freq: FxHashMap<String, usize> = FxHashMap::default();
    let mut term_freqs: Vec<FxHashMap<String, usize>> = Vec::new();
    let mut doc_lengths: Vec<usize> = Vec::new();

    for doc in &docs {
        let mut tf: FxHashMap<String, usize> = FxHashMap::default();
        let mut seen: FxHashSet<String> = FxHashSet::default();
        let mut len = 0;

        for m in TOKEN_REGEX.find_iter(doc) {
            let token = m.as_str().to_lowercase();
            if token.len() > 1 {
                *tf.entry(token.clone()).or_insert(0) += 1;
                len += 1;
                if !seen.contains(&token) {
                    *doc_freq.entry(token.clone()).or_insert(0) += 1;
                    seen.insert(token);
                }
            }
        }
        term_freqs.push(tf);
        doc_lengths.push(len);
    }

    let avg_doc_len: f32 = doc_lengths.iter().sum::<usize>() as f32 / docs.len() as f32;
    let num_docs = docs.len() as f32;

    c.bench_function("bm25_query_1000_docs", |bencher| {
        let query = "rust programming machine learning";
        let query_tokens: Vec<String> = TOKEN_REGEX
            .find_iter(query)
            .map(|m| m.as_str().to_lowercase())
            .filter(|s| s.len() > 1)
            .collect();

        bencher.iter(|| {
            let mut scores = vec![0.0f32; docs.len()];

            for token in &query_tokens {
                let df = *doc_freq.get(token).unwrap_or(&0) as f32;
                if df == 0.0 { continue; }

                let idf = ((num_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

                for (doc_id, tf_map) in term_freqs.iter().enumerate() {
                    let tf = *tf_map.get(token).unwrap_or(&0) as f32;
                    if tf == 0.0 { continue; }

                    let doc_len = doc_lengths[doc_id] as f32;
                    let k1 = 1.2f32;
                    let b = 0.75f32;
                    let norm = 1.0 - b + b * (doc_len / avg_doc_len);
                    let score = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
                    scores[doc_id] += score;
                }
            }
            black_box(scores)
        });
    });
}

/// Benchmark text chunking
fn bench_chunking(c: &mut Criterion) {
    let text = "fn main() {\n    println!(\"Hello, world!\");\n}\n\n".repeat(100);

    c.bench_function("simple_chunk_10kb", |bencher| {
        bencher.iter(|| {
            let chunk_size = 1000; // chars
            let overlap = 200;
            let mut chunks = Vec::new();
            let mut start = 0;

            while start < text.len() {
                let end = (start + chunk_size).min(text.len());
                chunks.push(&text[start..end]);
                if end >= text.len() { break; }
                start = end - overlap;
            }
            black_box(chunks)
        });
    });
}

/// Benchmark vector normalization
fn bench_vector_normalize(c: &mut Criterion) {
    let dims = 1536;
    let vec: Vec<f32> = (0..dims).map(|i| (i as f32) / 1000.0).collect();

    c.bench_function("normalize_1536d", |bencher| {
        bencher.iter(|| {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = vec.iter().map(|x| x / norm).collect();
            black_box(normalized)
        });
    });
}

/// Benchmark top-k selection using BinaryHeap (optimal for top-k)
fn bench_top_k_selection(c: &mut Criterion) {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    // Wrapper for f32 to implement Ord (min-heap for top-k max)
    #[derive(PartialEq)]
    struct MinScore(usize, f32);

    impl Eq for MinScore {}

    impl PartialOrd for MinScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for MinScore {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse order for min-heap behavior
            other.1.partial_cmp(&self.1).unwrap_or(Ordering::Equal)
        }
    }

    let mut group = c.benchmark_group("top_k");

    for size in [1000, 10000, 100000].iter() {
        let scores: Vec<(usize, f32)> = (0..*size)
            .map(|i| (i, (i as f32) / (*size as f32)))
            .collect();

        // Full sort approach
        group.bench_with_input(BenchmarkId::new("full_sort", size), size, |b, _| {
            b.iter(|| {
                let mut sorted = scores.clone();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sorted.truncate(10);
                black_box(sorted)
            });
        });

        // BinaryHeap approach (optimal for top-k)
        group.bench_with_input(BenchmarkId::new("heap", size), size, |b, _| {
            b.iter(|| {
                let k = 10;
                let mut heap: BinaryHeap<MinScore> = BinaryHeap::with_capacity(k + 1);

                for &(idx, score) in &scores {
                    heap.push(MinScore(idx, score));
                    if heap.len() > k {
                        heap.pop();
                    }
                }

                let result: Vec<(usize, f32)> = heap.into_sorted_vec()
                    .into_iter()
                    .map(|MinScore(idx, score)| (idx, score))
                    .collect();
                black_box(result)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_tokenization,
    bench_bm25_build,
    bench_bm25_query,
    bench_chunking,
    bench_vector_normalize,
    bench_top_k_selection,
);

criterion_main!(benches);
