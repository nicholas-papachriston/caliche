//! Restore a saved transcript into a live API session.

use anyhow::Result;
use caliche::harness::FittedResumeContext;
use caliche::messages::Message;

use super::{ApiAdapter, ApiAdapterError};

impl ApiAdapter {
    /// Load stored messages into the session the next turn sends to the model.
    ///
    /// # Errors
    ///
    /// Returns [`ApiAdapterError::SessionNotFound`] when `session_id` is not live,
    /// or an error when the transcript cannot be stored in session memory.
    pub async fn restore_resume_context(
        &self,
        session_id: &str,
        messages: Vec<Message>,
    ) -> Result<FittedResumeContext> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;
        let mut state = session.write().await;
        state.touch();
        state
            .harness
            .write()
            .await
            .restore_resume_context(messages)
            .await
    }
}
