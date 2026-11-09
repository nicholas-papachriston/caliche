//! Opaque cursor pagination for MCP list endpoints.
//!
//! Cursors are deterministic, versioned, and encode an offset into a
//! previously sorted list. Invalid cursors fail closed with a typed error.

use crate::protocol::Cursor;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use thiserror::Error;

/// Default page size for tools / prompts / resources lists.
pub const DEFAULT_PAGE_SIZE: usize = 64;

/// Maximum page size accepted from a client override.
pub const MAX_PAGE_SIZE: usize = 256;

const CURSOR_VERSION: u8 = 1;

/// Errors from cursor decode / pagination.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PaginationError {
    #[error("invalid pagination cursor")]
    InvalidCursor,
    #[error("page size must be between 1 and {MAX_PAGE_SIZE}")]
    InvalidPageSize,
}

/// One page of items plus an optional continuation cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<Cursor>,
}

/// Decode a cursor into a zero-based offset.
pub fn decode_cursor(cursor: Option<&str>) -> Result<usize, PaginationError> {
    let Some(raw) = cursor.filter(|c| !c.is_empty()) else {
        return Ok(0);
    };
    let bytes = URL_SAFE_NO_PAD
        .decode(raw.as_bytes())
        .map_err(|_| PaginationError::InvalidCursor)?;
    if bytes.len() != 9 || bytes[0] != CURSOR_VERSION {
        return Err(PaginationError::InvalidCursor);
    }
    let mut offset_bytes = [0u8; 8];
    offset_bytes.copy_from_slice(&bytes[1..]);
    Ok(usize::try_from(u64::from_be_bytes(offset_bytes))
        .map_err(|_| PaginationError::InvalidCursor)?)
}

/// Encode a zero-based offset as an opaque cursor.
pub fn encode_cursor(offset: usize) -> Cursor {
    let mut bytes = [0u8; 9];
    bytes[0] = CURSOR_VERSION;
    bytes[1..].copy_from_slice(&(offset as u64).to_be_bytes());
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Page a pre-sorted slice.
///
/// `page_size` of `None` uses [`DEFAULT_PAGE_SIZE`]. The caller must sort
/// `items` first so cursors remain stable across calls.
pub fn paginate<T: Clone>(
    items: &[T],
    cursor: Option<&str>,
    page_size: Option<usize>,
) -> Result<Page<T>, PaginationError> {
    let size = match page_size {
        None => DEFAULT_PAGE_SIZE,
        Some(0) => return Err(PaginationError::InvalidPageSize),
        Some(n) if n > MAX_PAGE_SIZE => return Err(PaginationError::InvalidPageSize),
        Some(n) => n,
    };
    let start = decode_cursor(cursor)?;
    if start > items.len() {
        return Err(PaginationError::InvalidCursor);
    }
    let end = (start + size).min(items.len());
    let page_items = items[start..end].to_vec();
    let next_cursor = if end < items.len() {
        Some(encode_cursor(end))
    } else {
        None
    };
    Ok(Page {
        items: page_items,
        next_cursor,
    })
}

/// Page a pre-sorted owned vector without cloning page items.
///
/// Moves the selected page out of `items` and drops the remainder. Prefer this
/// on hot list paths where the full roster was built only to serve one page.
pub fn paginate_owned<T>(
    mut items: Vec<T>,
    cursor: Option<&str>,
    page_size: Option<usize>,
) -> Result<Page<T>, PaginationError> {
    let size = match page_size {
        None => DEFAULT_PAGE_SIZE,
        Some(0) => return Err(PaginationError::InvalidPageSize),
        Some(n) if n > MAX_PAGE_SIZE => return Err(PaginationError::InvalidPageSize),
        Some(n) => n,
    };
    let start = decode_cursor(cursor)?;
    if start > items.len() {
        return Err(PaginationError::InvalidCursor);
    }
    let end = (start + size).min(items.len());
    let next_cursor = if end < items.len() {
        Some(encode_cursor(end))
    } else {
        None
    };
    items.truncate(end);
    let page_items = if start == 0 {
        items
    } else {
        items.drain(start..).collect()
    };
    Ok(Page {
        items: page_items,
        next_cursor,
    })
}

/// Sort key helper for tool / prompt names (byte-wise, case-sensitive).
pub fn cmp_name(a: &str, b: &str) -> std::cmp::Ordering {
    a.cmp(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty_cursor_starts_at_zero() {
        assert_eq!(decode_cursor(None).expect("ok"), 0);
        assert_eq!(decode_cursor(Some("")).expect("ok"), 0);
    }

    #[test]
    fn roundtrip_offset_cursor() {
        let encoded = encode_cursor(128);
        assert_eq!(decode_cursor(Some(&encoded)).expect("ok"), 128);
    }

    #[test]
    fn rejects_garbage_cursor() {
        assert_eq!(
            decode_cursor(Some("!!!")),
            Err(PaginationError::InvalidCursor)
        );
    }

    #[test]
    fn paginate_splits_deterministically() {
        let items: Vec<_> = (0..100).collect();
        let first = paginate(&items, None, Some(10)).expect("page");
        assert_eq!(first.items, (0..10).collect::<Vec<_>>());
        let cursor = first.next_cursor.expect("more");
        let second = paginate(&items, Some(&cursor), Some(10)).expect("page");
        assert_eq!(second.items, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn last_page_has_no_cursor() {
        let items = vec![1, 2, 3];
        let page = paginate(&items, None, Some(10)).expect("page");
        assert_eq!(page.items, items);
        assert!(page.next_cursor.is_none());
    }

    #[test]
    fn paginate_owned_moves_without_clone() {
        let items: Vec<_> = (0..5).map(|n| format!("t{n}")).collect();
        let page = paginate_owned(items, None, Some(2)).expect("page");
        assert_eq!(page.items, ["t0", "t1"]);
        assert!(page.next_cursor.is_some());
    }

    #[test]
    fn rejects_zero_page_size() {
        let items = [1, 2, 3];
        assert_eq!(
            paginate(&items, None, Some(0)).unwrap_err(),
            PaginationError::InvalidPageSize
        );
    }
}
