//! A provider's image URL is untrusted: it may not reach a private host.

use std::sync::{Arc, Mutex};

use caliche::llm::{
    ImageGenerationProvider, ImageGenerationRequest, OpenAIImageGenerationProvider,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn a_provider_image_url_on_loopback_is_not_fetched() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let asked = Arc::new(Mutex::new(Vec::<String>::new()));
    let log = Arc::clone(&asked);
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = vec![0_u8; 8192];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
            log.lock().expect("log").push(path.clone());
            let body = format!(r#"{{"data":[{{"url":"http://127.0.0.1:{port}/secret"}}]}}"#);
            let reply = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    let provider = OpenAIImageGenerationProvider::new("test-key", "image-model")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"));

    let err = provider
        .generate(ImageGenerationRequest {
            prompt: "a cat".to_string(),
            model: None,
            size: None,
            quality: None,
            output_format: None,
            count: None,
        })
        .await
        .expect_err("a loopback image URL must be refused");

    assert!(
        err.to_string().starts_with("OpenAI image URL refused: "),
        "{err}"
    );
    assert_eq!(
        *asked.lock().expect("log"),
        vec!["/v1/images/generations".to_string()]
    );
}
