pub mod codec;
pub mod index;
pub mod maxscore;
pub mod postings;
pub mod types;

pub use codec::{
    DurableDocumentMetaV1, DurableLexiconEntryV1, DurablePostingsBlockV1, FullTextCodecVersion,
};
pub use index::{FullTextIndex, FullTextIndexConfig, InMemoryFullTextIndex};
pub use types::{
    DocId, FullTextHit, LexiconEntry, Posting, PostingBlock, TermId, TextDocumentMeta,
};
