# Exploration: Semantic Search with Swift on Code Text Files

**Date:** 2026-02-17
**Branch:** `claude/swift-semantic-search-Qgdwd`
**Status:** Research / Feasibility exploration

## Summary

**Yes, it is feasible to build semantic search over code text files using Swift.** The Swift/Apple ecosystem now has mature-enough building blocks for on-device text embeddings, vector indexing, and even AST-aware code chunking. There are multiple viable approaches ranging from zero-dependency Apple-native APIs to flexible open-source packages.

---

## Available Building Blocks

### 1. Embedding Generation

#### Option A: Apple NLContextualEmbedding (macOS 14+ / iOS 17+)
- BERT-based contextual embeddings built into the OS
- 512-dimensional vectors, 256-token sequence length
- Produces per-token embeddings; requires mean-pooling for sentence/chunk vectors
- Zero external dependencies — ships with the OS
- **Limitation:** Models are organized by script/language (natural language, not programming language); model assets may need to be downloaded on first use; no control over architecture or fine-tuning
- Wrapped by the [NaturalLanguageEmbeddings](https://github.com/buh/NaturalLanguageEmbeddings) Swift package

#### Option B: swift-embeddings (jkrukowski/swift-embeddings)
- Open-source Swift package using `MLTensor` for local inference
- Supports loading models from Hugging Face: BERT, RoBERTa, XLM-RoBERTa, ModernBERT, MiniLM
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim) works well for code — compact and fast
- Full control over model choice and tokenization
- **Link:** https://github.com/jkrukowski/swift-embeddings

#### Option C: SimilaritySearchKit (ZachNagengast/similarity-search-kit)
- All-in-one Swift package: embeddings + indexing + search
- Ships with CoreML-converted models: DistilBERT, MiniLM variants
- Also supports Apple's native `NLEmbedding` as a backend
- Pre-converted CoreML models on Hugging Face
- **Link:** https://github.com/ZachNagengast/similarity-search-kit

#### Option D: VecturaKit
- Swift-based vector database for on-device RAG
- Supports multiple embedding backends: swift-embeddings, MLX, NLContextualEmbedding
- Includes hybrid search (vector + BM25)
- Model2Vec support with 32M parameter default model
- **Link:** https://github.com/rryam/VecturaKit

### 2. Vector Indexing & Search

| Package | Algorithm | Swift Native | Notes |
|---------|-----------|-------------|-------|
| [USearch](https://github.com/unum-cloud/USearch) | HNSW | Yes (Swift bindings) | Same engine leann-rs uses via Rust. Scales to 100M+ vectors on iPhone. User-defined metrics. |
| [similarity-topology](https://github.com/JadenGeller/similarity-topology) | HNSW | Pure Swift | LMDB-backed durable option. No deletion support. |
| [SimilaritySearchKit](https://github.com/ZachNagengast/similarity-search-kit) | Brute-force + metrics | Pure Swift | Cosine, dot product, Euclidean. Good for <100K vectors. |
| [VecturaKit](https://github.com/rryam/VecturaKit) | Vector DB | Pure Swift | Built-in BM25 hybrid search, pluggable storage backends. |
| [SimSIMD](https://github.com/ashvardanian/SimSIMD) | Distance metrics | Swift bindings | SIMD-accelerated similarity computation. Up to 200x faster. |

### 3. Code Parsing & Chunking

For semantic search over **code** (not prose), the chunking strategy matters. Options:

| Approach | Tool | Swift Support | Notes |
|----------|------|--------------|-------|
| AST-aware chunking | SwiftSyntax | Native Swift | Parses Swift source into AST; extract functions, types, comments |
| AST-aware chunking | Tree-sitter | C library + Swift bindings | Supports 15+ languages (Python, JS, Rust, Go, etc.) |
| Line/token-based chunking | Custom | Trivial in Swift | Simple but loses structural context |
| Markdown parsing | swift-markdown | Native Swift | For documentation files |

**Key insight:** The "Semly" project (jano.dev) already demonstrated a working Swift semantic code search tool using SwiftSyntax + MiniLM L6 v2 for on-device code search. It parses Swift and Markdown, generates embeddings, and performs similarity search — all locally.

---

## Architecture Options

### Option 1: Minimal — Apple Native Only
```
Code Files → Simple Chunker → NLContextualEmbedding → Cosine Similarity (Accelerate/vDSP)
```
- **Pros:** Zero dependencies, ships with macOS
- **Cons:** No HNSW index (brute-force only), limited to Apple's model, no AST-awareness
- **Best for:** Small codebases (<1K files), prototype/proof-of-concept

### Option 2: Balanced — Open-Source Swift Packages
```
Code Files → SwiftSyntax/Tree-sitter → swift-embeddings (MiniLM) → USearch (HNSW)
```
- **Pros:** Good model choice, fast HNSW search, AST-aware chunking
- **Cons:** More dependencies, requires model download
- **Best for:** Medium codebases, production-quality local search

### Option 3: Full-Featured — VecturaKit
```
Code Files → Tree-sitter → VecturaKit (embeddings + vector DB + BM25 hybrid search)
```
- **Pros:** Hybrid search out of the box, pluggable backends, closest to what leann-rs does
- **Cons:** Newer project (2025), fewer battle-tested deployments
- **Best for:** RAG-style code Q&A applications

---

## Relevance to leann-rs

leann-rs already has AST-aware chunking (`src/chunker/ast.rs`) and uses USearch for HNSW indexing. A Swift port or companion tool could:

1. **Reuse the USearch index format** — USearch indexes are cross-language compatible, so a Swift client could read indexes built by leann-rs
2. **Leverage the same architecture** — chunking → embedding → HNSW search → hybrid BM25 blend
3. **Run fully on-device** on macOS/iOS without needing a Python runtime or external API calls
4. **Use Tree-sitter** for multi-language AST parsing (same approach as leann-rs's AST chunker)

---

## Key Considerations & Limitations

### What works well
- **On-device embedding** is mature on Apple platforms (NLContextualEmbedding, CoreML, swift-embeddings)
- **Vector search** has multiple Swift-compatible options, including USearch which leann-rs already uses
- **Swift-specific code parsing** is excellent via SwiftSyntax
- **Multi-language code parsing** is possible via Tree-sitter Swift bindings
- **Performance** is sufficient — MiniLM runs in milliseconds per chunk on Apple Silicon

### What to watch out for
- **Embedding model quality for code:** General-purpose models (MiniLM, BERT) work for code but are not fine-tuned for it. Code-specific models (CodeBERT, StarEncoder) would need CoreML conversion
- **NLContextualEmbedding limitations:** Token-level output requires pooling; model assets may need downloading; trained on natural language not code
- **Platform lock-in:** Apple-native APIs require macOS 14+ / iOS 17+; open-source packages are more portable but still primarily target Apple platforms
- **Index size:** For very large codebases, HNSW indexes can be memory-intensive. USearch handles this well but is still a consideration on mobile
- **No code-specific embedding models in Swift ecosystem yet:** You'd need to convert a model like `microsoft/codebert-base` to CoreML using `coremltools` in Python first

---

## Conclusion

Building semantic search over code files in Swift is **feasible and practical today**. The recommended starting point would be:

1. Use **swift-embeddings** with `all-MiniLM-L6-v2` for generating code chunk embeddings
2. Use **SwiftSyntax** (for Swift) or **Tree-sitter** (for multi-language) for AST-aware chunking
3. Use **USearch** for HNSW vector indexing (cross-compatible with leann-rs indexes)
4. Optionally add **BM25** keyword search for hybrid retrieval

This would closely mirror leann-rs's architecture but run natively on Apple platforms without Rust or Python dependencies.

---

## References

- [NaturalLanguageEmbeddings](https://github.com/buh/NaturalLanguageEmbeddings) — Swift wrapper for NLContextualEmbedding
- [SimilaritySearchKit](https://github.com/ZachNagengast/similarity-search-kit) — On-device embeddings + search
- [VecturaKit](https://github.com/rryam/VecturaKit) — Swift vector database for on-device RAG
- [swift-embeddings](https://github.com/jkrukowski/swift-embeddings) — Run embedding models locally via MLTensor
- [USearch](https://github.com/unum-cloud/USearch) — HNSW vector search with Swift bindings
- [similarity-topology](https://github.com/JadenGeller/similarity-topology) — Pure Swift HNSW
- [SwiftSemanticSearch](https://github.com/ashvardanian/SwiftSemanticSearch) — USearch + UForm demo app
- [SimSIMD](https://github.com/ashvardanian/SimSIMD) — SIMD-accelerated similarity metrics
- [NLContextualEmbedding docs](https://developer.apple.com/documentation/naturallanguage/nlcontextualembedding)
- [NLEmbedding docs](https://developer.apple.com/documentation/naturallanguage/nlembedding)
- [Semly — Swift semantic code search](https://www.jano.dev/swift/2025/09/28/semantic-search.html)
- [Semantic Code Search with AST chunking](https://wangxj03.github.io/posts/2024-09-24-code-search/)
