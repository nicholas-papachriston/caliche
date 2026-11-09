//! `documents` API client methods.

use anyhow::Result;

use super::ApiClient;
use crate::api::types::{
    ChainResponse, CollectionsResponse, DocumentIndexRequest, DocumentIndexResponse,
    DocumentMapReduceRequest, DocumentSearchRequest, DocumentSearchResponse, DocumentSplitRequest,
    DocumentSplitResponse, LLMChainRequest, MapReduceRequest, RetrievalChainRequest,
};

impl ApiClient {
    // ========================================================================
    // Document / RAG Methods
    // ========================================================================

    /// Split documents into chunks
    pub async fn split_documents(
        &self,
        request: &DocumentSplitRequest,
    ) -> Result<DocumentSplitResponse> {
        self.rpc_call("documents.split", request).await
    }

    /// Index documents into a collection
    pub async fn index_documents(
        &self,
        request: &DocumentIndexRequest,
    ) -> Result<DocumentIndexResponse> {
        self.rpc_call("documents.index", request).await
    }

    /// List document collections
    pub async fn list_collections(&self) -> Result<CollectionsResponse> {
        match self
            .rpc_call("documents.listCollections", &serde_json::json!({}))
            .await
        {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "documents.listCollections") {
                    Ok(CollectionsResponse {
                        collections: Vec::new(),
                    })
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Delete a document collection
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                "documents.deleteCollection",
                &serde_json::json!({ "name": name }),
            )
            .await?;
        Ok(())
    }

    /// Search documents in a collection
    pub async fn search_documents(
        &self,
        request: &DocumentSearchRequest,
    ) -> Result<DocumentSearchResponse> {
        self.rpc_call("documents.search", request).await
    }

    // ========================================================================
    // Chain Methods
    // ========================================================================

    pub async fn chains_llm(&self, request: &LLMChainRequest) -> Result<ChainResponse> {
        self.rpc_call("chains.llm", request).await
    }

    pub async fn chains_map_reduce(&self, request: &MapReduceRequest) -> Result<ChainResponse> {
        self.rpc_call("chains.mapReduce", request).await
    }

    pub async fn chains_retrieval(&self, request: &RetrievalChainRequest) -> Result<ChainResponse> {
        self.rpc_call("chains.retrieval", request).await
    }

    pub async fn chains_document_map_reduce(
        &self,
        request: &DocumentMapReduceRequest,
    ) -> Result<ChainResponse> {
        self.rpc_call("chains.documentMapReduce", request).await
    }
}
