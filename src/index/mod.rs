//! Index module - index building, searching, and metadata

mod meta;
mod passages;
mod builder;
mod searcher;
mod filter;
mod bm25;
mod embeddings;
mod recompute;
mod locate;
mod query;

pub use meta::IndexMeta;
pub use builder::{IndexBuilder, StreamingIndexBuilder};
pub use passages::{Passage, PassageStore};
pub use searcher::{IndexSearcher, SearchOptions, SearchResult};
pub use filter::MetadataFilter;
pub use embeddings::{EmbeddingsStore, prune_embeddings};
pub use recompute::RecomputeSearcher;
pub use locate::find_index;
pub use query::{expand_from_passages, should_expand};
