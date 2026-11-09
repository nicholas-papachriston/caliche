import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const contracts = require("./index.cjs");

export default contracts;
export const parseContextRefKind = contracts.parseContextRefKind;
export const parsePermissionMode = contracts.parsePermissionMode;
export const parseSlashIntent = contracts.parseSlashIntent;
