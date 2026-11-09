/**
 * Local inference catalog fit example.
 *
 * Probes host RAM/VRAM and prints which built-in catalog models are safe
 * to load.
 *
 * Usage:
 *   bun run examples/local_inference_fit.ts
 */

import {
  fitLocalInferenceCatalog,
  probeLocalInferenceHost,
  safeLocalInferenceModels,
} from "../src/index";

function main(): void {
  if (!probeLocalInferenceHost || !fitLocalInferenceCatalog || !safeLocalInferenceModels) {
    console.error(
      "Local inference fit APIs unavailable. Rebuild with the local-inference feature."
    );
    process.exitCode = 1;
    return;
  }

  const host = probeLocalInferenceHost();
  console.log("Host capabilities:");
  console.log(`  RAM available: ${host.availableRamBytes} / ${host.totalRamBytes}`);
  if (host.availableVramBytes != null) {
    console.log(`  VRAM free: ${host.availableVramBytes}`);
  }
  console.log(`  Recommended device: ${host.recommendedDevice}`);
  for (const note of host.notes) {
    console.log(`  Note: ${note}`);
  }

  console.log("\nCatalog fit:");
  for (const fit of fitLocalInferenceCatalog()) {
    console.log(
      `  [${fit.status}] ${fit.modelId}  n_ctx=${fit.recommendedNCtx} max_safe=${fit.maxSafeNCtx}`
    );
  }

  const safe = safeLocalInferenceModels();
  console.log(`\nSafe models (${safe.length}): ${safe.map((m) => m.modelId).join(", ")}`);
}

main();
