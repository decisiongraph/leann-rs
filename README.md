# LEANN-RS

Rust implementation of LEANN - Lightweight Embedding-based Approximate Nearest Neighbor search.

A single-binary CLI for building, searching, and querying vector indexes for RAG applications.

## Features

- **Single binary** - No Python runtime, no dependencies
- **Fast** - Native Rust with optimized HNSW/DiskANN search
- **Portable** - Cross-platform (Linux, macOS, Windows)
- **Compatible** - Reads Python LEANN index format
- **Hybrid search** - Combine vector + BM25 keyword search
- **Metadata filtering** - Filter results by document attributes
- **ReAct agent** - Multi-turn reasoning with tool use
- **HTTP server** - REST API for integration

## Installation

```bash
# Build from source
cargo build --release

# Binary at target/release/leann (~4MB)
```

## Quick Start

```bash
# Build an index from documents
leann build my-docs --docs ./documents --embedding-mode ollama --embedding-model nomic-embed-text

# Search
leann search my-docs "How does authentication work?"

# Ask questions (RAG)
leann ask my-docs "Explain the architecture" --llm ollama --model qwen3:8b

# Interactive mode
leann ask my-docs --interactive
```

## Commands

### Build Index

```bash
# Basic build with OpenAI embeddings
leann build my-docs --docs ./documents

# Use Ollama for local embeddings
leann build my-docs --docs ./documents --embedding-mode ollama --embedding-model nomic-embed-text

# Custom options
leann build my-code --docs ./src \
  --file-types ".rs,.py,.ts" \
  --doc-chunk-size 512 \
  --graph-degree 48
```

### Search

```bash
# Basic search
leann search my-docs "vector database"

# With metadata filtering
leann search my-docs "authentication" -f "source:*.rs"

# Hybrid search (vector + BM25)
leann search my-docs "user login" --hybrid

# JSON output
leann search my-docs "query" --format json
```

### Ask (RAG)

```bash
# Single question
leann ask my-docs "How does caching work?"

# Use different LLM providers
leann ask my-docs "question" --llm openai --model gpt-4o
leann ask my-docs "question" --llm anthropic --model claude-3-5-sonnet-20241022
leann ask my-docs "question" --llm ollama --model qwen3:8b

# Interactive chat
leann ask my-docs --interactive
```

### ReAct Agent

Multi-turn reasoning with search tool:

```bash
leann react my-docs "What are all the ways errors are handled?"

# Show reasoning trace
leann react my-docs "Compare feature X and Y" --verbose --max-steps 10
```

### HTTP Server

```bash
# Start server
leann serve my-docs --port 8080 --cors

# API endpoints:
# POST /search - Search the index
# GET  /info   - Index information
# GET  /health - Health check
```

### Manage Indexes

```bash
# List all indexes
leann list

# Detailed info
leann list --detailed

# Remove an index
leann remove my-docs
```

## Optional Features

```bash
# Build with PDF support
cargo build --release --features pdf

# Build with HTTP server
cargo build --release --features server

# Build with DiskANN backend
cargo build --release --features diskann-backend

# Build with all features
cargo build --release --features full
```

## Search Options

### Metadata Filtering

Filter by document metadata:

```bash
# By file extension
leann search my-docs "query" -f "source:*.rs"

# By type
leann search my-docs "query" -f "type=code"

# Numeric comparison
leann search my-docs "query" -f "lines>100"
```

Supported operators: `=`, `!=`, `>`, `>=`, `<`, `<=`, `:` (glob patterns)

### Hybrid Search

Combine vector similarity with BM25 keyword matching:

```bash
leann search my-docs "exact function name" --hybrid --hybrid-alpha 0.5
```

Alpha controls the balance: 1.0 = pure vector, 0.0 = pure BM25.

## Index Compatibility

LEANN-RS reads Python LEANN indexes:
- `.passages.jsonl` - Text passages
- `.passages.idx.json` - JSON offset map
- `.index` - Vector index (usearch format)
- `.meta.json` - Metadata
- `.ids.txt` - ID mapping

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible URL |
| `ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `OLLAMA_HOST` | Ollama server URL (default: http://localhost:11434) |

## Binary Sizes

| Build | Size |
|-------|------|
| Default | ~4 MB |
| + PDF | ~5 MB |
| + Server | ~5 MB |
| + DiskANN | ~6 MB |
| Full | ~8 MB |

## Architecture

```
src/
├── cli/           # Commands (build, search, ask, react, serve)
├── index/         # Index management, BM25, filtering
├── backend/       # HNSW (usearch), DiskANN
├── embedding/     # OpenAI, Ollama providers
└── llm/           # OpenAI, Ollama, Anthropic
```

## License

MIT
