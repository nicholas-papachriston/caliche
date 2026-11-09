//! Object Storage Example
//!
//! This example demonstrates:
//! - Cloudflare R2 configuration
//! - MinIO local setup
//! - Generic S3-compatible storage
//! - Environment-based provider switching
//! - Programmatic credential passing (long-term and temporary credentials)
//!
//! Best practices shown:
//! - Using same API across different providers
//! - Environment-driven configuration
//! - Passing credentials via code (more secure than env vars)
//! - Using temporary credentials with session tokens
//!
//! Run with:
//! ```bash
//! cargo run --example object_storage --features s3
//! ```
//!
//! Prerequisites:
//! - For Cloudflare R2: Set CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY
//! - For MinIO: Set OBJECT_STORE_ENDPOINT=http://localhost:9000, OBJECT_STORE_PATH_STYLE=true
//! - For AWS S3: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (optional, can pass via code)

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use bytes::Bytes;
#[cfg(feature = "s3")]
use caliche::storage::{ObjectStoreConfig, create_object_store};

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(not(feature = "s3"))]
    {
        println!("This example requires the 's3' feature");
        return Ok(());
    }

    #[cfg(feature = "s3")]
    {
        dotenvy::dotenv().ok();

        // Example 1: Auto-detect from environment
        let _config = ObjectStoreConfig::from_env()?;
        println!("Using object store provider from environment");

        // Example 2: Explicit Cloudflare R2 configuration
        if let (Ok(account_id), Ok(access_key), Ok(secret_key)) = (
            std::env::var("CLOUDFLARE_ACCOUNT_ID"),
            std::env::var("CLOUDFLARE_ACCESS_KEY_ID"),
            std::env::var("CLOUDFLARE_SECRET_ACCESS_KEY"),
        ) {
            let r2_config =
                ObjectStoreConfig::cloudflare_r2(account_id, access_key, secret_key, None);
            let r2_store = create_object_store(r2_config).await?;

            let data = Bytes::from("Hello from Cloudflare R2!");
            r2_store
                .put_object("my-bucket", "test/r2-example.txt", data, Some("text/plain"))
                .await?;

            println!("Uploaded to Cloudflare R2");
        }

        // Example 3: Generic S3-compatible (MinIO) with programmatic credentials
        if let Ok(endpoint) = std::env::var("OBJECT_STORE_ENDPOINT") {
            let minio_config = ObjectStoreConfig::generic_s3(
                endpoint,
                "us-east-1".to_string(),
                true, // path_style for MinIO
                Some("minioadmin".to_string()),
                Some("minioadmin".to_string()),
                None, // No session token for MinIO
            );
            let minio_store = create_object_store(minio_config).await?;

            let data = Bytes::from("Hello from MinIO!");
            minio_store
                .put_object(
                    "test-bucket",
                    "test/minio-example.txt",
                    data,
                    Some("text/plain"),
                )
                .await?;

            println!("Uploaded to MinIO");
        }

        // Example 4: AWS S3 with long-term credentials (programmatic)
        println!("\nExample: AWS S3 with programmatic credentials");
        println!("  Long-term credentials:");
        let _aws_config_long_term = ObjectStoreConfig::aws_s3(
            Some("us-east-1".to_string()),
            Some("your-access-key-id".to_string()),
            Some("your-secret-access-key".to_string()),
            None, // No session token for long-term credentials
        );
        println!("    Config created with access_key_id and secret_access_key");

        // Example 5: AWS S3 with temporary credentials (programmatic)
        println!("  Temporary credentials (from STS/IAM roles):");
        let _aws_config_temp = ObjectStoreConfig::aws_s3(
            Some("us-east-1".to_string()),
            Some("temporary-access-key-id".to_string()),
            Some("temporary-secret-access-key".to_string()),
            Some("session-token-from-sts".to_string()), // Session token for temporary credentials
        );
        println!("    Config created with access_key_id, secret_access_key, and session_token");
        println!("    Note: Temporary credentials expire and should be refreshed periodically");

        // Example 6: Security best practices
        println!("\nSecurity Best Practices:");
        println!("  - Pass credentials via code (not env vars) for better security");
        println!("  - Use temporary credentials (session tokens) when possible");
        println!("  - Never commit credentials to version control");
        println!("  - Rotate credentials regularly");
        println!("  - Use IAM roles for applications running on AWS infrastructure");

        // Example 7: Same API works with agent tools
        println!("\nAgent tools automatically use configured object store:");
        println!("  fs works with s3:// URIs");
        println!("  process_document works with s3:// URIs");
        println!("  All tools use the same provider configured via environment or code");
    }

    Ok(())
}
