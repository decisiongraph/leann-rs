//! Index module - index building, searching, and metadata

mod meta;
mod passages;
mod builder;
mod searcher;
mod filter;
mod bm25;
mod embeddings;
mod recompute;

pub use meta::IndexMeta;
pub use builder::IndexBuilder;
pub use passages::{Passage, PassageStore};
pub use searcher::{IndexSearcher, SearchOptions, SearchResult};
pub use filter::MetadataFilter;
pub use embeddings::{EmbeddingsStore, EmbeddingsWriter, prune_embeddings};
pub use recompute::RecomputeSearcher;
