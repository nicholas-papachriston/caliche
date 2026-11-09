//! Shared AWS helpers used by managed-vector backends.
//!
//! Gated by `s3-backend` and/or `dynamodb-backend` so lean builds stay small.

#[cfg(any(feature = "s3-backend", feature = "dynamodb-backend"))]
pub(crate) mod sigv4_http;

#[cfg(feature = "dynamodb-backend")]
pub(crate) use sigv4_http::send_signed_json_value;
#[cfg(any(feature = "s3-backend", feature = "dynamodb-backend"))]
pub(crate) use sigv4_http::{
    AwsSdkConfigOpts, load_sdk_config, send_signed_json, sign_json_http_request,
};
