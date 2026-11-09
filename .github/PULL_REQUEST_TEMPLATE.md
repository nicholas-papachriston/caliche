<!--
Title: a Conventional Commit header, for example `fix(tools): handle empty patch`.
See CONTRIBUTING.md and AGENTS.md.
-->

## Summary

<!-- What changed and why. -->

Fixes #
<!-- Use `Part of #N` instead when this PR does not finish the issue. -->

## Gate

Run from the repository root, in order (the agent gate in AGENTS.md):

- [ ] `./caliche rust fmt-check && ./caliche rust clippy`
- [ ] `./caliche check`
- [ ] `./caliche test`
- [ ] `./caliche build-dev`

Also run when relevant:

- [ ] `./caliche ci` (wasm-opt check, fmt-check, Clippy, complexity gate, Rust tests, JS typecheck, and JS and Python tests)
- [ ] `./caliche rust test`
- [ ] `./caliche rust complexity --gate`

<!-- Paste any command you could not run and why. -->

## shotput submodule

- [ ] No shotput submodule pointer change
- [ ] shotput submodule pointer bump: the shotput commit is merged and pushed to the shotput repository (link the shotput PR)

## Dependencies

- [ ] No new dependencies
- [ ] New dependency added. Justification (why existing crates or packages do not cover it):

## Notes

<!-- Follow-ups, risks, or anything reviewers should check by hand. -->
