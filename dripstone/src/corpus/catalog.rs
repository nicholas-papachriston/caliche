//! Named corpora and where their bytes come from.
//!
//! English Cohere shards live under `en/`. Simple English is a different tree
//! and is not a default.

use crate::core::Metric;
use crate::errors::DripstoneError;

/// ann-benchmarks NYTimes angular vectors (256-D).
pub const NYTIMES_URL: &str = "http://ann-benchmarks.com/nytimes-256-angular.hdf5";
/// ann-benchmarks SIFT Euclidean vectors (128-D).
pub const SIFT_URL: &str = "http://ann-benchmarks.com/sift-128-euclidean.hdf5";
/// ann-benchmarks GIST Euclidean vectors (960-D, about 3.6 GB).
pub const GIST_URL: &str = "http://ann-benchmarks.com/gist-960-euclidean.hdf5";

/// Hugging Face dataset repo for Cohere Wikipedia embeddings.
pub const COHERE_REPO: &str = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3";
/// Language prefix inside that repo. English, not Simple English.
pub const COHERE_LANG_PREFIX: &str = "en";
/// Hugging Face revision for [`COHERE_REPO`].
pub const COHERE_REVISION: &str = "main";
/// Cohere embed-multilingual-v3 width.
pub const COHERE_DIMENSION: usize = 1024;

/// Hugging Face dataset for the CoIR CoSQA code-retrieval corpus.
pub const COSQA_REPO: &str = "CoIR-Retrieval/cosqa";
/// Pinned revision of [`COSQA_REPO`]. The layout directory changes with this pin.
pub const COSQA_REVISION: &str = "0846fa3b963a21bead36e9fab61451fc83b777a6";
/// Corpus parquet inside [`COSQA_REPO`].
pub const COSQA_CORPUS_FILE: &str = "corpus/corpus-00000-of-00001.parquet";
/// Query parquet inside [`COSQA_REPO`].
pub const COSQA_QUERIES_FILE: &str = "queries/queries-00000-of-00001.parquet";
/// Test qrels parquet inside [`COSQA_REPO`].
pub const COSQA_QRELS_FILE: &str = "data/test-00000-of-00001.parquet";
/// CoSQA code is Python. Per-row `language` still wins when a row says otherwise.
pub const COSQA_LANGUAGE: &str = "python";
/// Width of the hashed code features stored for [`CorpusName::Code`].
pub const CODE_FEATURE_DIMENSION: usize = 256;

const ANN_REVISION: &str = "ann-benchmarks";
const COHERE_LAYOUT_REVISION: &str = "2023-11";
const COSQA_LAYOUT_REVISION: &str = "cosqa-0846fa3b";

/// How a normalized corpus was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusFormat {
    /// ann-benchmarks HDF5 (`train`, `test`, optional `neighbors`).
    Hdf5,
    /// Parquet files with a list/fixed-size-list embedding column.
    Parquet,
    /// CoIR CoSQA code and queries, split with tree-sitter and hashed to features.
    Code,
}

/// Catalog id. Repeated names stay on this enum instead of loose strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusName {
    /// `nytimes-256-angular`.
    Nytimes256Angular,
    /// `sift-128-euclidean`.
    Sift128Euclidean,
    /// `gist-960-euclidean`. Resume the HDF5 download; it is several gigabytes.
    Gist960Euclidean,
    /// Cohere Wikipedia embeddings, English `en/` Parquet shards.
    CohereWikiEn,
    /// CoIR CoSQA (`CoIR-Retrieval/cosqa`). Feature vectors, cosine.
    Code,
}

impl CorpusName {
    /// Stable directory name under `corpora/`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nytimes256Angular => "nytimes-256-angular",
            Self::Sift128Euclidean => "sift-128-euclidean",
            Self::Gist960Euclidean => "gist-960-euclidean",
            Self::CohereWikiEn => "cohere-wiki-en",
            Self::Code => "code",
        }
    }

    /// Layout revision directory.
    #[must_use]
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Nytimes256Angular | Self::Sift128Euclidean | Self::Gist960Euclidean => {
                ANN_REVISION
            }
            Self::CohereWikiEn => COHERE_LAYOUT_REVISION,
            Self::Code => COSQA_LAYOUT_REVISION,
        }
    }

    /// On-disk format for a loadable entry.
    #[must_use]
    pub const fn format(self) -> CorpusFormat {
        match self {
            Self::Nytimes256Angular | Self::Sift128Euclidean | Self::Gist960Euclidean => {
                CorpusFormat::Hdf5
            }
            Self::CohereWikiEn => CorpusFormat::Parquet,
            Self::Code => CorpusFormat::Code,
        }
    }

    /// Distance the publisher used for neighbors.
    #[must_use]
    pub const fn metric(self) -> Metric {
        match self {
            Self::Nytimes256Angular => Metric::Cosine,
            Self::Sift128Euclidean | Self::Gist960Euclidean => Metric::Euclidean,
            Self::CohereWikiEn => Metric::DotProduct,
            Self::Code => Metric::Cosine,
        }
    }

    /// Expected vector width, when the catalog knows it.
    #[must_use]
    pub const fn dimension(self) -> Option<usize> {
        match self {
            Self::Nytimes256Angular => Some(256),
            Self::Sift128Euclidean => Some(128),
            Self::Gist960Euclidean => Some(960),
            Self::CohereWikiEn => Some(COHERE_DIMENSION),
            Self::Code => Some(CODE_FEATURE_DIMENSION),
        }
    }

    /// Single-file download URL. Parquet corpora use a shard helper instead.
    #[must_use]
    pub const fn download_url(self) -> Option<&'static str> {
        match self {
            Self::Nytimes256Angular => Some(NYTIMES_URL),
            Self::Sift128Euclidean => Some(SIFT_URL),
            Self::Gist960Euclidean => Some(GIST_URL),
            Self::CohereWikiEn | Self::Code => None,
        }
    }

    /// Disk-preflight hint for the source object. Not an integrity check.
    #[must_use]
    pub const fn size_hint_bytes(self) -> Option<u64> {
        match self {
            // About 3.6 GB. Reserve a little more so a resume is not started
            // on a volume that cannot hold the file.
            Self::Gist960Euclidean => Some(4 * 1024 * 1024 * 1024),
            // Corpus, queries, and test qrels together are a few megabytes.
            Self::Code => Some(8 * 1024 * 1024),
            Self::Nytimes256Angular | Self::Sift128Euclidean | Self::CohereWikiEn => None,
        }
    }

    /// Language shard directory inside a multilingual dataset.
    #[must_use]
    pub const fn language(self) -> Option<&'static str> {
        match self {
            Self::CohereWikiEn => Some(COHERE_LANG_PREFIX),
            Self::Code => Some(COSQA_LANGUAGE),
            Self::Nytimes256Angular | Self::Sift128Euclidean | Self::Gist960Euclidean => None,
        }
    }

    /// Parse a catalog name. Simple English is rejected on purpose.
    ///
    /// # Errors
    ///
    /// Returns an error when `name` is empty, unknown, or selects Simple English.
    pub fn parse(name: &str) -> Result<Self, DripstoneError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "nytimes-256-angular" | "nytimes" => Ok(Self::Nytimes256Angular),
            "sift-128-euclidean" | "sift" => Ok(Self::Sift128Euclidean),
            "gist-960-euclidean" | "gist" => Ok(Self::Gist960Euclidean),
            "cohere-wiki-en" | "cohere-wiki" | "cohere-en" | "wikipedia-2023-11-en" => {
                Ok(Self::CohereWikiEn)
            }
            "code" | "code-corpus" | "cosqa" | "coir-cosqa" | "coir" => Ok(Self::Code),
            "simple" | "cohere-simple" | "simple-english" | "wikipedia-2023-11-simple" => {
                Err(DripstoneError::storage(
                    "Simple English is not a Dripstone corpus default; use `cohere-wiki-en` (shards under en/)",
                ))
            }
            other => Err(DripstoneError::storage(format!(
                "unknown corpus `{other}`; expected nytimes-256-angular, sift-128-euclidean, gist-960-euclidean, cohere-wiki-en, or code"
            ))),
        }
    }

    /// Match an ann-benchmarks URL to a catalog entry, ignoring scheme.
    #[must_use]
    pub fn from_url(url: &str) -> Option<Self> {
        const CANDIDATES: [CorpusName; 3] = [
            CorpusName::Nytimes256Angular,
            CorpusName::Sift128Euclidean,
            CorpusName::Gist960Euclidean,
        ];
        CANDIDATES.into_iter().find(|name| {
            name.download_url()
                .is_some_and(|known| urls_match(known, url))
        })
    }
}

/// Where bytes are read from before normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusSelector {
    /// A built-in catalog entry.
    Catalog(CorpusName),
    /// HDF5 file at an explicit URL (integration override).
    Hdf5Url {
        /// Directory and manifest name.
        name: String,
        /// Download URL.
        url: String,
    },
    /// Already-staged Parquet file or directory.
    ParquetDir {
        /// Directory and manifest name.
        name: String,
        /// File or directory of `.parquet` shards.
        path: std::path::PathBuf,
    },
    /// Staged CoIR CoSQA tree. Same files the downloader writes under `source/`.
    CodeDir {
        /// Directory and manifest name.
        name: String,
        /// Directory of corpus, queries, and optional qrels.
        path: std::path::PathBuf,
    },
}

fn urls_match(left: &str, right: &str) -> bool {
    fn host_path(value: &str) -> String {
        value
            .trim()
            .trim_end_matches('/')
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .to_ascii_lowercase()
    }
    host_path(left) == host_path(right)
}
