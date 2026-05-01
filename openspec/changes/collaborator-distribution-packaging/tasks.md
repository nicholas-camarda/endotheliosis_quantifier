## 1. Governance alignment

- [x] 1.1 Cross-read `openspec/changes/label-studio-medsam-hybrid-grading/{proposal,design}.md` and ensure no contradictory packaging claims vs Stage 1 model-blind grading non-goals.

## 2. Documentation + illustrative stubs

- [x] 2.1 Update `README.md` collaborator/bootstrap sections to reference Releases-first + `EQ_RUNTIME_ROOT` vs Git-only misconceptions (link OpenSpec capability name textually).
- [x] 2.2 Add focused collaborator doc (prefer extending `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` or add `docs/COLLABORATOR_DISTRIBUTION.md`) summarizing topology: LS UI vs companion vs `eq` quant workflows.
- [x] 2.3 Confirm `docs/examples/artifacts_manifest.example.json` matches final manifest schema decisions; extend roles/fields if audit narrows naming.

## 3. Compose packaging (documentation-first)

- [x] 3.1 Draft git-tracked Compose snippet directory (`deploy/compose/` or agreed path per design `[defer_ok]`) documenting LS + companion volume mounts without committing weights.

## 4. Validation

- [ ] 4.1 Run `openspec validate collaborator-distribution-packaging --strict`.
- [ ] 4.2 If README paths change materially, smoke-check internal links.
