"use strict";

const adapter = require("@caliche/core/adapters/ai-sdk");

module.exports = {
  CalicheLanguageModel: adapter.CalicheLanguageModel,
  CalicheMcpBridge: adapter.CalicheMcpBridge,
  caliche: adapter.caliche,
};
