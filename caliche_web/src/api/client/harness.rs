//! External CLI harness registry client.

use anyhow::Result;

use super::ApiClient;
use crate::api::types::{
    CreateHarnessRequest, DispatchHarnessRequest, DispatchHarnessResponse, HarnessInfo,
    HarnessListResponse,
};

impl ApiClient {
    /// List registered CLI harnesses.
    pub async fn list_harnesses(&self) -> Result<Vec<HarnessInfo>> {
        let resp: HarnessListResponse = self
            .rpc_call("orchestration.listHarnesses", &serde_json::json!({}))
            .await?;
        Ok(resp.harnesses)
    }

    /// Register a CLI harness.
    pub async fn create_harness(&self, req: CreateHarnessRequest) -> Result<HarnessInfo> {
        self.rpc_call("orchestration.createHarness", &req).await
    }

    /// Remove a registered harness.
    pub async fn delete_harness(&self, id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                "orchestration.deleteHarness",
                &serde_json::json!({ "id": id }),
            )
            .await?;
        Ok(())
    }

    /// Run one task on a registered harness.
    pub async fn dispatch_harness(
        &self,
        id: &str,
        message: &str,
    ) -> Result<DispatchHarnessResponse> {
        let req = DispatchHarnessRequest {
            message: message.to_string(),
        };
        self.rpc_call(
            "orchestration.dispatchHarness",
            &serde_json::json!({ "id": id, "message": req.message }),
        )
        .await
    }
}
