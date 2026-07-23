//! Error type for hdmseg.

use std::fmt;

/// Errors returned by the segmentation pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdmError {
    /// The input stack had an invalid shape (needs `N >= 1`, `M >= 2`,
    /// `D in {2, 3}`, and a coordinate buffer of length `N*M*D`).
    InvalidStack(String),
    /// A configuration value was out of range.
    InvalidConfig(String),
    /// The symmetric eigensolver failed to converge.
    EigenFailed,
    /// A requested number of clusters/components exceeded what the data
    /// supports.
    TooManyClusters { requested: usize, available: usize },
}

impl fmt::Display for HdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdmError::InvalidStack(m) => write!(f, "invalid stack: {m}"),
            HdmError::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            HdmError::EigenFailed => write!(f, "symmetric eigensolver failed to converge"),
            HdmError::TooManyClusters {
                requested,
                available,
            } => write!(
                f,
                "requested {requested} clusters but only {available} loci/components available"
            ),
        }
    }
}

impl std::error::Error for HdmError {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, HdmError>;
