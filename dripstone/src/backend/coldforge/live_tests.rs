use super::{
    LiveAwsOptions, LiveAwsPlan, LiveBackend, SceneRow, cold_search_samples, format_tsv,
    parse_backend_name, percentile, plan_from_env, recall_at_k, render,
};

#[test]
fn percentile_uses_nearest_rank() {
    let samples = [10.0, 20.0, 30.0, 40.0];
    assert!((percentile(&samples, 0.50) - 20.0).abs() < f64::EPSILON);
    assert!((percentile(&samples, 0.90) - 40.0).abs() < f64::EPSILON);
}

#[test]
fn backend_names_are_the_three_scene_backends() {
    assert_eq!(
        parse_backend_name("minio").expect("minio"),
        LiveBackend::Minio
    );
    assert_eq!(parse_backend_name("s3").expect("s3"), LiveBackend::S3);
    assert_eq!(
        parse_backend_name("s3-redis").expect("redis"),
        LiveBackend::S3Redis
    );
    assert!(parse_backend_name("valkey").is_err());
    assert!(parse_backend_name("dynamodb").is_err());
    assert!(parse_backend_name("s3-vectors").is_err());
}

#[test]
fn empty_search_recall_is_zero() {
    let relevant = vec!["row-0#0".to_string()];
    assert!((recall_at_k(&[], &relevant)).abs() < f64::EPSILON);
    assert!((recall_at_k(&["row-0#0".to_string()], &relevant) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn tsv_header_is_one_scene_table() {
    let row = SceneRow {
        backend: "s3".to_string(),
        n: 50,
        upsert_flush_ms: 1.0,
        search_cold_p50_ms: 2.0,
        search_cold_p90_ms: 3.0,
        search_cold_p99_ms: 4.0,
        search_warm_p50_ms: 5.0,
        search_warm_p90_ms: 6.0,
        search_warm_p99_ms: 7.0,
        get_p50_ms: 8.0,
        get_p90_ms: 9.0,
        head_bucket_p50_ms: None,
        hits: 10,
        brute_hits: 10,
        recall_at_k: 1.0,
        answer_at_k: None,
        support_at_k: Some(0.5),
    };
    let tsv = format_tsv(&[row]);
    let header = tsv.lines().next().expect("header");
    assert!(header.contains("upsert_flush_ms"));
    assert!(header.contains("search_cold_p99_ms"));
    assert!(header.contains("head_bucket_p50_ms"));
    assert!(header.contains("recall_at_k"));
    assert!(header.contains("answer_at_k"));
    assert!(header.contains("support_at_k"));
    let body = tsv.lines().nth(1).expect("row");
    assert!(body.starts_with("s3\t50\t"));
    assert!(body.ends_with("\t1.000\t-\t0.500"));
}

#[test]
fn aws_cold_samples_stay_small_and_minio_can_stress() {
    assert_eq!(cold_search_samples(LiveBackend::S3), 2);
    assert_eq!(cold_search_samples(LiveBackend::S3Redis), 2);
    assert_eq!(cold_search_samples(LiveBackend::Minio), 50);
    assert!(cold_search_samples(LiveBackend::S3) < cold_search_samples(LiveBackend::Minio));
}

#[test]
fn report_documents_cold_split_refresh_and_honest_gets() {
    let options = LiveAwsOptions {
        backend: LiveBackend::S3,
        bucket: "example-bucket".to_string(),
        region: "us-east-1".to_string(),
        key_prefix: "coldforge-live/1/".to_string(),
        redis_url: None,
    };
    let markdown = render(
        &options,
        &super::super::live_corpus::synthetic_header(),
        &[],
        "backend\n",
        "HeadBucket skipped.",
    );
    assert!(markdown.contains("cold_queries: 2\n"));
    assert!(markdown.contains("warm_queries: 50\n"));
    assert!(markdown.contains("wal_refresh_interval_ms: 1000\n"));
    assert!(markdown.contains("drops the in-process body map"));
    assert!(markdown.contains("2 on s3 and s3-redis, 50 on minio"));
    assert!(markdown.contains("scene metric"));
    assert!(!markdown.contains("60000"));
    let minio = LiveAwsOptions {
        backend: LiveBackend::Minio,
        ..options
    };
    let minio_report = render(
        &minio,
        &super::super::live_corpus::synthetic_header(),
        &[],
        "backend\n",
        "HeadBucket skipped.",
    );
    assert!(minio_report.contains("cold_queries: 50\n"));
}

#[test]
fn missing_live_flag_skips_without_numbers() {
    if std::env::var("DRIPSTONE_LIVE_AWS").ok().as_deref() == Some("1") {
        return;
    }
    match plan_from_env().expect("plan") {
        LiveAwsPlan::Skip { reason } => assert!(reason.contains("DRIPSTONE_LIVE_AWS")),
        LiveAwsPlan::Run(_) => panic!("unset flag must skip"),
    }
}
