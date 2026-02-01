//! HTTP utilities for API providers

use std::time::Duration;

use reqwest::{Client, Response};

/// Create a reqwest client with connection pooling and sensible defaults
///
/// The client is configured with:
/// - Connection pooling (max 10 idle connections per host)
/// - 30 second timeout
/// - Keepalive connections
pub fn create_client() -> Client {
    Client::builder()
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

/// Check HTTP response status and return detailed error if not successful
///
/// This helper extracts error details from the response body for better debugging.
pub async fn check_response(response: Response, service_name: &str) -> anyhow::Result<Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    // Try to extract error message from JSON response
    let error_detail = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
        // Common API error formats
        json.get("error")
            .and_then(|e| e.get("message").and_then(|m| m.as_str()))
            .or_else(|| json.get("message").and_then(|m| m.as_str()))
            .or_else(|| json.get("detail").and_then(|d| d.as_str()))
            .map(|s| s.to_string())
            .unwrap_or(body)
    } else {
        body
    };

    anyhow::bail!("{} API error {}: {}", service_name, status, error_detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_client() {
        // Just verify it creates without panicking
        let _client = create_client();
    }
}
