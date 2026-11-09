/// Failure while spawning a peer or speaking ACP v1 to it.
#[derive(Debug, thiserror::Error)]
pub enum AcpClientError {
    /// `argv` was empty, or the program name was empty.
    #[error("ACP argv must include a program")]
    EmptyArgv,

    /// [`crate::AcpClient::spawn`] was called outside a Tokio runtime.
    #[error("ACP client must run inside a Tokio runtime")]
    OutsideRuntime,

    /// The operating system rejected the child process.
    #[error("failed to spawn ACP agent `{program}`: {source}")]
    Spawn {
        /// `argv[0]` as lossy text, for the error string only.
        program: String,
        /// OS error from process spawn.
        #[source]
        source: std::io::Error,
    },

    /// The child was created without the stdio pipe the client needs.
    #[error("ACP agent {stream} pipe is missing")]
    StdioUnavailable {
        /// `stdin`, `stdout`, or `stderr`.
        stream: &'static str,
    },

    /// `cwd` was an empty path.
    #[error("ACP cwd must not be empty")]
    EmptyCwd,

    /// `cwd` could not be made absolute.
    #[error("failed to resolve the ACP working directory: {source}")]
    WorkingDirectory {
        /// OS error from `current_dir`.
        #[source]
        source: std::io::Error,
    },

    /// ACP JSON is UTF-8, so paths that are not Unicode cannot be sent.
    #[error("ACP paths must be valid UTF-8")]
    PathNotUtf8,

    /// A request frame could not be encoded.
    #[error("failed to encode ACP {what}: {source}")]
    Encode {
        /// Which frame failed.
        what: &'static str,
        /// Serde error.
        #[source]
        source: serde_json::Error,
    },

    /// A response body did not match the ACP type.
    #[error("failed to decode ACP {what}: {source}")]
    Decode {
        /// Which body failed.
        what: &'static str,
        /// Serde error.
        #[source]
        source: serde_json::Error,
    },

    /// `session/new`, `session/prompt`, or `session/cancel` ran before a successful `initialize`.
    #[error("ACP initialize has not completed")]
    NotInitialized,

    /// The child was killed, or `initialize` rejected the peer.
    #[error("ACP client is shut down")]
    ShutDown,

    /// Two `session/prompt` calls overlapped on one client.
    #[error("a session/prompt is already in flight")]
    PromptInFlight,

    /// `sessionId` was an empty string.
    #[error("ACP session id must not be empty")]
    EmptySessionId,

    /// The stdout pipe closed before the matching response arrived.
    #[error("ACP agent stdout closed before the response arrived")]
    StdoutClosed,

    /// A JSON-RPC request waited longer than the caller-configured limit.
    #[error("ACP {method} timed out")]
    TimedOut {
        /// JSON-RPC method that did not answer in time.
        method: String,
    },

    /// A stdout line was not one JSON object.
    ///
    /// Child stdout is the ACP pipe. A log line here means the peer (or something
    /// writing into that pipe) broke the protocol.
    #[error("ACP stdout is not protocol JSON: {detail}")]
    Protocol {
        /// Truncated offending line, or a short reason.
        detail: String,
    },

    /// The peer returned a JSON-RPC error object.
    #[error("ACP JSON-RPC error {code}: {message}")]
    Rpc {
        /// JSON-RPC error code.
        code: i32,
        /// JSON-RPC error message.
        message: String,
    },

    /// The peer selected a `protocolVersion` other than 1.
    #[error(
        "peer protocolVersion {offered} is not supported; Caliche speaks ACP version {supported}"
    )]
    UnsupportedProtocol {
        /// Version from the peer's `initialize` result.
        offered: u32,
        /// The only version this client speaks.
        supported: u32,
    },

    /// The peer advertised an auth method outside the hint allowlist.
    ///
    /// An empty list, or a list of only `opencode-login` and `hermes-setup`,
    /// does not produce this error. The client has not called `authenticate`.
    #[error("{message}")]
    AuthRejected {
        /// Operator-facing reason. Includes the method ids when the peer sent them.
        message: String,
    },

    /// The peer returned ACP `auth_required` after `initialize`.
    ///
    /// Provider API keys and local models must be configured on that peer
    /// outside Caliche. This client does not call `authenticate` and does
    /// not launch a terminal login.
    #[error("{message}")]
    AuthRequired {
        /// Operator-facing BYO guidance, including the peer's error text.
        message: String,
    },

    /// Read or write on the child pipes failed.
    #[error("ACP agent IO failed: {source}")]
    Io {
        /// OS error from the pipe.
        #[source]
        source: std::io::Error,
    },

    /// The JSONL transcript could not be created or updated.
    #[error("failed to write the ACP transcript: {source}")]
    Transcript {
        /// OS error from the transcript file.
        #[source]
        source: std::io::Error,
    },
}
