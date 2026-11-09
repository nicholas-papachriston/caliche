"use strict";

const mcp = require("@caliche/core/mcp");
const sse = require("@caliche/core/transport/sse");

module.exports = {
  LATEST_PROTOCOL_VERSION: mcp.LATEST_PROTOCOL_VERSION,
  SUPPORTED_PROTOCOL_VERSIONS: mcp.SUPPORTED_PROTOCOL_VERSIONS,
  McpServerBuilder: mcp.McpServerBuilder,
  McpServer: mcp.McpServer,
  InProcessMcpServerBuilder: mcp.InProcessMcpServerBuilder,
  InProcessMcpServer: mcp.InProcessMcpServer,
  encodeSseEvent: sse.encodeSseEvent,
  decodeSseChunk: sse.decodeSseChunk,
  decodeSseStream: sse.decodeSseStream,
};
