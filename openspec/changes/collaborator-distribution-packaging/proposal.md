## Why

Collaborators need a clear, documented way to obtain **code + configs + heavyweight artifacts** without assuming everything lives in Git or that **Label Studio replaces `eq` segmentation/quantification**. Recent discussions surfaced confusion between **GitHub**, **Git LFS**, **runtime roots**, **Docker Compose**, and **what runs where**. Separately, **`label-studio-medsam-hybrid-grading`** plans a **MedSAM companion** for box-assist and preload—but **does not** currently adopt **burden quantification model predictions inside primary Label Studio grading** (that remains `eq` workflows / exports per existing grading contracts).

## What Changes

- Introduce an explicit **`collaborator-distribution-packaging`** capability covering **canonical distribution channels**, **artifact manifests**, **topology options** (conda-only vs Compose companion bundle), and **boundaries**: LS UI vs MedSAM companion vs quantification pipelines.
- Document **recommended primary channel**: **versioned GitHub Release bundles** (checksum + provenance stub) with optional **Git LFS** only for a **small pinned allowlist** when bandwidth/storage trade-offs are acceptable—without reversing the repo mandate that bulk runtime trees stay under **`EQ_RUNTIME_ROOT`**.
- Clarify relationship to **`label-studio-medsam-hybrid-grading`**: hybrid uses **MedSAM companion HTTP** + mask releases; **quantification scoring models** remain **out of Stage 1 model-blind grading UX** unless a **future explicit OpenSpec change** relaxes that contract.
- Extend collaborator-facing documentation pointers under **`label-studio-local-bootstrap`** requirements so bootstrap runbooks reference the distribution manifest story.

## Capabilities

### New Capabilities

- `collaborator-distribution-packaging`: Canonical collaborator artifact channels, artifact manifest expectations, Compose-vs-local topology guidance, separation of LS labeling vs `eq` quantification execution, cross-references to hybrid MedSAM companion—not vending servers inside `src/eq` unless a later change states otherwise.

### Modified Capabilities

- `label-studio-local-bootstrap`: ADD normative pointers tying collaborator bootstrap docs to the distribution/manifest contract (no behavioral change required on day one beyond documentation/tests if applicable).

## Impact

- Documentation (`README.md`, `docs/`), possibly `configs/` templates for collaborator YAML examples and **`docs/examples/artifacts_manifest.example.json`** (git-tracked stub).
- Cross-links from **`openspec/changes/label-studio-medsam-hybrid-grading/`** proposal/design (reference only; implementation sequencing stays dependency-ordered).
- No mandatory move of existing checkpoints into Git; governance wording reconciles historical `.gitattributes` LFS patterns with **runtime-first** defaults.

## Explicit Decisions

- **Distribution narrative owner**: OpenSpec capability **`collaborator-distribution-packaging`** plus tracked docs—not informal chat-only guidance.
- **Primary documented artifact channel**: **GitHub Releases** for collaborator-facing pinned bundles; **Git LFS** documented as **optional** secondary channel with explicit quota/size caveats.
- **Quantification inside Label Studio**: **Not in scope** for this change’s collaborator labeling story; **`label-studio-medsam-hybrid-grading`** explicitly lists **excluding Stage 2 grade suggestions from Stage 1 primary grading UX**. Any **assistive quant preview inside LS** requires a **separate OpenSpec change** that revisits model-blind grading contracts.

## Open Questions

- `[defer_ok]` Exact **`artifacts_manifest.json`** schema version and SHA algorithm naming (`sha256` required vs optional secondary hashes).
- `[defer_ok]` Whether v1 ships **Docker Compose** only as documented snippets versus git-tracked `deploy/compose/` files—topology guidance belongs in design/tasks regardless.
