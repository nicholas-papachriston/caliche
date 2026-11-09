//! Bootstrap standard S3 bucket and S3 Vector bucket/index for integration benchmarks.

use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::types::{BucketLocationConstraint, CreateBucketConfiguration};
use aws_sdk_s3vectors::types::{
    DataType, DistanceMetric as S3vDistanceMetric, Index, MetadataConfiguration,
};
use aws_smithy_runtime_api::client::result::SdkError;
use dripstone::backend::s3::vectors::VectorDistanceMetric;
use dripstone::{DripstoneError, Metric};

pub struct IntegrationAwsConfig {
    pub region: String,
    /// Region for S3 Vector Buckets control plane (must be a [vectors-supported region](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-regions-quotas.html)).
    pub s3_vectors_region: String,
    pub s3_bucket: String,
    /// Standard S3 bucket in `s3_vectors_region` for `__dripstone_indexes/...` (must match `S3VectorBackend` SDK region).
    pub s3_vector_sidecar_bucket: String,
    pub s3_prefix: String,
    pub s3_vector_bucket: String,
    pub s3_vector_index: String,
}

fn env_or(name: &str, default: impl FnOnce() -> String) -> String {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(default)
}

/// Fail before any AWS call when the integration opt-in buckets are unset.
pub fn require_integration_opt_in() -> Result<(), DripstoneError> {
    for name in ["DRIPSTONE_INT_S3_BUCKET", "DRIPSTONE_INT_S3_VECTOR_BUCKET"] {
        let present = std::env::var(name)
            .ok()
            .is_some_and(|value| !value.trim().is_empty());
        if !present {
            return Err(DripstoneError::storage(format!(
                "{name} must be set to run AWS integration tests"
            )));
        }
    }
    Ok(())
}

/// Returns `true` if default credential chain can call STS.
pub async fn probe_credentials() -> bool {
    let cfg = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let sts = aws_sdk_sts::Client::new(&cfg);
    sts.get_caller_identity().send().await.is_ok()
}

fn bucket_location_config(
    region: &str,
) -> Result<Option<CreateBucketConfiguration>, DripstoneError> {
    if region == "us-east-1" {
        return Ok(None);
    }
    let constraint = BucketLocationConstraint::from(region);
    let cbc = CreateBucketConfiguration::builder()
        .location_constraint(constraint)
        .build();
    Ok(Some(cbc))
}

/// Resolve bucket names from env or derive from account + region.
pub async fn load_integration_aws_config() -> Result<IntegrationAwsConfig, DripstoneError> {
    require_integration_opt_in()?;
    let region = env_or("AWS_REGION", || "us-east-1".to_string());

    let cfg = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(region.clone()))
        .load()
        .await;

    let sts = aws_sdk_sts::Client::new(&cfg);
    let ident = sts
        .get_caller_identity()
        .send()
        .await
        .map_err(|e| DripstoneError::storage(format!("STS get_caller_identity: {e}")))?;
    let account = ident
        .account()
        .ok_or_else(|| DripstoneError::storage("STS response missing account".to_string()))?
        .to_string();

    let s3_vectors_region = env_or("DRIPSTONE_INT_S3_VECTORS_REGION", || region.clone());

    let s3_bucket = env_or("DRIPSTONE_INT_S3_BUCKET", || {
        format!("dripstone-int-{account}-{region}")
    });
    let s3_vector_bucket = env_or("DRIPSTONE_INT_S3_VECTOR_BUCKET", || {
        format!("dripstone-vb-{account}-{s3_vectors_region}")
    });
    let s3_vector_index = env_or("DRIPSTONE_INT_S3_VECTOR_INDEX", || {
        "dripstone-bench-index".to_string()
    });
    let s3_prefix = env_or("DRIPSTONE_INT_S3_PREFIX", || "coldforge/bench/".to_string());

    let s3_vector_sidecar_bucket = env_or("DRIPSTONE_INT_S3_VECTOR_SIDECAR_BUCKET", || {
        if s3_vectors_region == region {
            s3_bucket.clone()
        } else {
            format!("dripstone-int-{account}-{s3_vectors_region}")
        }
    });

    Ok(IntegrationAwsConfig {
        region,
        s3_vectors_region,
        s3_bucket,
        s3_vector_sidecar_bucket,
        s3_prefix,
        s3_vector_bucket,
        s3_vector_index,
    })
}

fn s3vectors_distance_metric(m: Metric) -> S3vDistanceMetric {
    match VectorDistanceMetric::try_from_core_metric(m) {
        Ok(VectorDistanceMetric::Cosine) => S3vDistanceMetric::Cosine,
        Ok(VectorDistanceMetric::Euclidean) => S3vDistanceMetric::Euclidean,
        Err(_) => S3vDistanceMetric::Cosine,
    }
}

fn validate_s3vectors_metric(metric: Metric) -> Result<(), DripstoneError> {
    match VectorDistanceMetric::try_from_core_metric(metric) {
        Ok(_) => Ok(()),
        Err(err) => Err(err),
    }
}

fn s3_index_matches_corpus(index: &Index, dimension: i32, distance_metric: Metric) -> bool {
    index.dimension() == dimension
        && *index.distance_metric() == s3vectors_distance_metric(distance_metric)
}

fn format_s3vectors_sdk_error<E: std::fmt::Debug, R: std::fmt::Debug>(
    operation: &str,
    err: &SdkError<E, R>,
) -> String {
    match err {
        SdkError::ServiceError(ctx) => {
            format!("{operation}: {:?}", ctx.err())
        }
        _ => format!("{operation}: {err:?}"),
    }
}

fn s3_create_bucket_idempotent_ok(err_msg: &str) -> bool {
    // Same-account "you already created this bucket" after a prior run or racing callers.
    err_msg.contains("BucketAlreadyOwnedByYou")
}

/// Ensure a classic S3 bucket exists in the given AWS region (for object storage, not vector buckets).
pub async fn ensure_bucket_in_region(region: &str, bucket: &str) -> Result<(), DripstoneError> {
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(region.to_string()))
        .load()
        .await;
    let s3 = aws_sdk_s3::Client::new(&sdk);

    // HeadBucket is optional: some accounts/SDKs surface a generic "service error" for missing
    // buckets or strict IAM. If head succeeds, the bucket is usable.
    if s3.head_bucket().bucket(bucket).send().await.is_ok() {
        return Ok(());
    }

    let mut req = s3.create_bucket().bucket(bucket);
    if let Some(loc) = bucket_location_config(region)? {
        req = req.create_bucket_configuration(loc);
    }

    match req.send().await {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if s3_create_bucket_idempotent_ok(&msg) {
                Ok(())
            } else {
                Err(DripstoneError::storage(format!(
                    "create_bucket {bucket} (region {region}): {e}. \
If the bucket exists in another region, set AWS_REGION or override the bucket env for this region.",
                )))
            }
        }
    }
}

pub async fn ensure_standard_bucket(cfg: &IntegrationAwsConfig) -> Result<(), DripstoneError> {
    ensure_bucket_in_region(&cfg.region, &cfg.s3_bucket).await
}

/// Sidecar bucket for S3 Vector namespace index JSON; always in `s3_vectors_region`.
pub async fn ensure_s3_vector_sidecar_bucket(
    cfg: &IntegrationAwsConfig,
) -> Result<(), DripstoneError> {
    ensure_bucket_in_region(&cfg.s3_vectors_region, &cfg.s3_vector_sidecar_bucket).await
}

/// `Ok(true)` when the configured vector index already exists.
///
/// A not-found response is `Ok(false)`. This does not create a bucket or index.
pub async fn vector_index_present(cfg: &IntegrationAwsConfig) -> Result<bool, DripstoneError> {
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(cfg.s3_vectors_region.clone()))
        .load()
        .await;
    let client = aws_sdk_s3vectors::Client::new(&sdk);
    match client
        .get_index()
        .vector_bucket_name(cfg.s3_vector_bucket.clone())
        .index_name(cfg.s3_vector_index.clone())
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(err) => {
            let not_found = matches!(
                &err,
                SdkError::ServiceError(ctx) if ctx.err().is_not_found_exception()
            ) || err.to_string().to_ascii_lowercase().contains("not found");
            if not_found {
                Ok(false)
            } else {
                Err(DripstoneError::storage(format_s3vectors_sdk_error(
                    "get_index",
                    &err,
                )))
            }
        }
    }
}

pub async fn ensure_vector_bucket_and_index(
    cfg: &IntegrationAwsConfig,
    dimension: i32,
    distance_metric: Metric,
) -> Result<(), DripstoneError> {
    validate_s3vectors_metric(distance_metric)?;
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(cfg.s3_vectors_region.clone()))
        .load()
        .await;
    let v = aws_sdk_s3vectors::Client::new(&sdk);

    let vb = v
        .create_vector_bucket()
        .vector_bucket_name(cfg.s3_vector_bucket.clone())
        .send()
        .await;

    match vb {
        Ok(_) => {}
        Err(ref e) => {
            let conflict = matches!(
                e,
                SdkError::ServiceError(ctx) if ctx.err().is_conflict_exception()
            );
            if !conflict {
                return Err(DripstoneError::storage(format!(
                    "{}. S3 Vector Buckets are only available in select regions; try \
                     DRIPSTONE_INT_S3_VECTORS_REGION=us-east-1 (or set AWS_REGION to a supported region). \
                     See AWS docs: S3 user guide -> Regions and endpoints for S3 Vectors.",
                    format_s3vectors_sdk_error("create_vector_bucket", e)
                )));
            }
        }
    }

    let meta = MetadataConfiguration::builder()
        .non_filterable_metadata_keys("[]")
        .build()
        .map_err(|e| DripstoneError::storage(format!("MetadataConfiguration: {e}")))?;

    let get = v
        .get_index()
        .vector_bucket_name(cfg.s3_vector_bucket.clone())
        .index_name(cfg.s3_vector_index.clone())
        .send()
        .await;

    match get {
        Ok(ref output) => {
            if let Some(idx) = output.index() {
                if s3_index_matches_corpus(idx, dimension, distance_metric) {
                    return Ok(());
                }
                let del = v
                    .delete_index()
                    .vector_bucket_name(cfg.s3_vector_bucket.clone())
                    .index_name(cfg.s3_vector_index.clone())
                    .send()
                    .await;
                match del {
                    Ok(_) => {}
                    Err(ref e) => {
                        let not_found = matches!(
                            e,
                            SdkError::ServiceError(ctx) if ctx.err().is_not_found_exception()
                        );
                        if !not_found {
                            return Err(DripstoneError::storage(format_s3vectors_sdk_error(
                                "delete_index",
                                e,
                            )));
                        }
                    }
                }
            }
        }
        Err(ref e) => {
            let not_found = matches!(
                e,
                SdkError::ServiceError(ctx) if ctx.err().is_not_found_exception()
            );
            if !not_found {
                return Err(DripstoneError::storage(format_s3vectors_sdk_error(
                    "get_index",
                    e,
                )));
            }
        }
    }

    let ix = v
        .create_index()
        .vector_bucket_name(cfg.s3_vector_bucket.clone())
        .index_name(cfg.s3_vector_index.clone())
        .data_type(DataType::Float32)
        .dimension(dimension)
        .distance_metric(s3vectors_distance_metric(distance_metric))
        .metadata_configuration(meta)
        .send()
        .await;

    match ix {
        Ok(_) => Ok(()),
        Err(ref e) => {
            let conflict = matches!(
                e,
                SdkError::ServiceError(ctx) if ctx.err().is_conflict_exception()
            );
            if conflict {
                let verify = v
                    .get_index()
                    .vector_bucket_name(cfg.s3_vector_bucket.clone())
                    .index_name(cfg.s3_vector_index.clone())
                    .send()
                    .await
                    .map_err(|e2| {
                        DripstoneError::storage(format_s3vectors_sdk_error(
                            "get_index after create_index conflict",
                            &e2,
                        ))
                    })?;
                let idx = verify.index().ok_or_else(|| {
                    DripstoneError::storage(
                        "get_index after create_index conflict returned no index".to_string(),
                    )
                })?;
                if s3_index_matches_corpus(idx, dimension, distance_metric) {
                    Ok(())
                } else {
                    Err(DripstoneError::storage(format!(
                        "S3 vector index '{}' in bucket '{}' exists with dimension {} and metric {:?}, \
                         but this run expects dimension {} and metric {:?}. Delete the index or set \
                         DRIPSTONE_INT_S3_VECTOR_INDEX to a fresh name.",
                        cfg.s3_vector_index,
                        cfg.s3_vector_bucket,
                        idx.dimension(),
                        idx.distance_metric(),
                        dimension,
                        s3vectors_distance_metric(distance_metric),
                    )))
                }
            } else {
                Err(DripstoneError::storage(format_s3vectors_sdk_error(
                    "create_index",
                    e,
                )))
            }
        }
    }
}
