"use strict";

const builtinTools = require("@caliche/core/builtin-tools");
const hooks = require("@caliche/core/hooks");
const jsonlSession = require("@caliche/core/jsonl-session");
const mutationStyle = require("@caliche/core/mutation-style");
const session = require("@caliche/core/session");
const toolInspection = require("@caliche/core/tool-inspection");
const toolRegistry = require("@caliche/core/tool-registry");

module.exports = {
  AgentSession: session.AgentSession,
  HookMatcher: hooks.HookMatcher,
  PermissionMode: hooks.PermissionMode,
  MutationStyle: mutationStyle.MutationStyle,
  InspectionAction: toolInspection.InspectionAction,
  InspectionHookAdapter: toolInspection.InspectionHookAdapter,
  PathSecurityInspector: toolInspection.PathSecurityInspector,
  PermissionModeInspector: toolInspection.PermissionModeInspector,
  ToolInspectionManager: toolInspection.ToolInspectionManager,
  ToolRegistry: toolRegistry.ToolRegistry,
  JsonlSessionHeader: jsonlSession.JsonlSessionHeader,
  JsonlSessionEntry: jsonlSession.JsonlSessionEntry,
  JsonlSessionTree: jsonlSession.JsonlSessionTree,
  JsonlSessionStore: jsonlSession.JsonlSessionStore,
  defaultInspectionManager: toolInspection.defaultInspectionManager,
  parseJsonlSession: jsonlSession.parseJsonlSession,
  newEntryId: jsonlSession.newEntryId,
  createApplyPatchTool: mutationStyle.createApplyPatchTool,
  createCalculatorTool: builtinTools.createCalculatorTool,
  createFsTool: mutationStyle.createFsTool,
};
