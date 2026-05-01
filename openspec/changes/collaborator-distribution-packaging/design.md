## Context

The repo already separates **Git-tracked software** from **`EQ_RUNTIME_ROOT`** artifacts (`AGENTS.md`). **`eq labelstudio start`** bootstraps **Label Studio in Docker** for grading UI only. **`label-studio-medsam-hybrid-grading`** introduces YAML-bound **mask-release preload** and a **MedSAM companion** HTTP boundary for **box-assisted** proposals; it explicitly **does not** expose **Stage 2 quantification suggestions** into **Stage 1** grading UX.

Collaborators still need one coherent story: **clone repo**, **obtain weights/releases**, **run LS**, **run `eq` quantification workflows** where applicable—without believing GitHub alone hosts GPU stacks.

## Goals / Non-Goals

**Goals:**

- Publish **explicit distribution tiers**: code (Git), documented **Release bundles**, optional **LFS allowlist**, mandatory **runtime layout** for images/data not in Git.
- Describe **recommended Compose topology** (LS + companion) as packaging guidance aligned with hybrid grading `[defer_ok]` notes—implementation may trail docs when companion stabilizes.
- Record **scientific / UX boundary**: quantification models remain **`eq run-config` / burden workflows on exported grades**, not silently merged into LS unless a future contract change permits it.

**Non-Goals:**

- Hosting collaborator GPUs or datasets inside GitHub Actions by default.
- Implementing the MedSAM companion server inside `src/eq` (unless a later architectural change reverses hybrid grading’s boundary statement).
- Replacing **`label-studio-medsam-hybrid-grading`** tasks; this change **documents packaging alignment** and sequencing hints only.

## Decisions

### Decision: Releases-first for collaborator bundles

**Choice:** Treat **GitHub Release attachments + checksum manifest** as the primary documented path for collaborator weights smaller than institutional limits.

**Alternative:** Git LFS-only → simpler clone UX but brittle quotas/costs; acceptable only as **documented optional** channel.

### Decision: Compose-first ergonomics

**Choice:** Recommend **`docker compose`** bringing up **Label Studio** + **companion** with **volume-mounted `./weights/<bundle_id>/`** populated from Releases download—single mental model.

**Alternative:** Ad-hoc “everyone configures conda paths” → highest drift; retained as advanced path only.

### Decision: Quantification boundary stays explicit

**Choice:** Primary grading remains **human-first / model-blind Stage 1** per hybrid grading non-goals; quantification consumes **exported authoritative grades**.

**Alternative:** Inline burden predictor suggestions inside LS Stage 1 → violates stated hybrid grading contract without new OpenSpec governance.

### Decision: Manifest stub in-repo

**Choice:** Track **`docs/examples/artifacts_manifest.example.json`** (small, illustrative) referencing Release URLs placeholders—not live secrets.

## Risks / Trade-offs

- **[Risk]** Contributors confuse Git-tracked `.gitattributes` legacy patterns with mandatory LFS → **Mitigation**: docs emphasize runtime-first defaults; example manifest cites Releases.
- **[Risk]** Pickle artifacts distributed broadly → **Mitigation**: document trusted-lab stance; prefer torch-export-friendly artifacts where companion permits.
- **[Risk]** Compose drift vs pinned LS tag → **Mitigation**: hybrid grading already flags pinning LS images—reuse same guidance here.

## Migration Plan

1. Land docs + example manifest + OpenSpec deltas (no workflow behavior change required initially).
2. When hybrid companion lands, replace placeholder Compose snippets with tested files under agreed directory (`deploy/` or `docker/`—finalize in tasks).
3. Rollback is documentation-only revert.

## Explicit Decisions

- **Companion + LS topology guidance** lives alongside **`configs/label_studio_medsam_hybrid.yaml`** references once hybrid grading ships; this change supplies packaging framing first.
- **Quant assist inside LS** is **explicitly deferred** to a future OpenSpec unless grading contracts change.

## Open Questions

- [defer_ok] Exact directory name for Compose snippets (`deploy/compose/` vs `docker/`).
- [audit_first_then_decide] Institutional artifact mirrors: whether manifests MUST allow non-GitHub `download_urls` arrays—audit lab IT constraints and review evidence from deployment during implementation.
