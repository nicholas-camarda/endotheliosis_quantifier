# glomeruli-candidate-comparison Specification

## Purpose
Define the supported mitochondria-transfer-versus-no-base comparison workflow, deterministic promotion evidence, and explicit promotion decision contract for glomeruli segmentation artifacts.
## Requirements
### Requirement: Glomeruli promotion compares transfer and no-base candidates on the same evidence set
The repository SHALL compare at least one mitochondria-transfer glomeruli candidate and one no-mitochondria-base glomeruli candidate using the same deterministic promotion-evaluation evidence.

#### Scenario: Candidate-comparison run is started
- **WHEN** the glomeruli candidate-comparison workflow is executed
- **THEN** it trains or loads a canonical mitochondria-transfer candidate and a canonical no-mitochondria-base candidate under the supported `raw_data/cohorts` all-admitted-masked contract or an explicitly selected active paired project root
- **AND** the canonical initial workflow uses one explicit seed per candidate family
- **AND** both candidates are evaluated on the same deterministic validation manifest
- **AND** each manifest crop is assigned to exactly one review category rather than being reused across `background`, `boundary`, and `positive`
- **AND** the manifest selection prefers spanning multiple source images when enough qualifying evidence exists
- **AND** the workflow does not silently fall back from transfer to no-base training or from no-base training to transfer during the comparison run
- **AND** the no-base candidate records its encoder initialization as the ImageNet-pretrained ResNet34 baseline rather than implying literal all-random initialization

#### Scenario: One candidate family cannot be completed
- **WHEN** either the transfer or no-base candidate cannot be trained or evaluated successfully
- **THEN** the comparison report records that failure explicitly
- **AND** the workflow does not silently substitute a different candidate family

### Requirement: Candidate comparison produces a deterministic promotion report
The glomeruli candidate-comparison workflow SHALL write a promotion report artifact that makes the comparison and decision auditable.

#### Scenario: Promotion report is generated
- **WHEN** candidate comparison completes
- **THEN** the report records each candidate's provenance, deterministic-manifest metrics, trivial-baseline comparisons, and prediction-review results
- **AND** the report records the seed used for each candidate
- **AND** the report states the final decision outcome as `promoted`, `blocked`, or `insufficient_evidence`
- **AND** the report includes a manifest-coverage summary that records total crops plus unique-image and unique-subject coverage
- **AND** the HTML review surface labels each panel with category, crop provenance, per-panel metrics, and explicit panel-order semantics
- **AND** the report structure preserves a clean path for future repeated-seed candidate rows without breaking the initial artifact contract

#### Scenario: Shared-manifest prediction semantics are applied
- **WHEN** candidate probabilities are converted into binary segmentation masks for promotion comparison
- **THEN** the comparison workflow SHALL use learner-consistent preprocessing rather than a bespoke evaluation-only image transform
- **AND** it SHALL use the supported underconfident-model inference threshold semantics for the current segmentation path rather than assuming a generic `0.5` threshold

#### Scenario: Compatibility artifact is available
- **WHEN** a current compatibility-era glomeruli artifact is available for comparison
- **THEN** the promotion report includes it as a non-promoted comparison artifact alongside the transfer and scratch candidates

#### Scenario: Candidate family failures are reported
- **WHEN** a candidate family is unavailable and the workflow still emits a structured promotion report
- **THEN** that report SHALL record the family failure explicitly
- **AND** report generation alone SHALL NOT be treated as evidence that the supported transfer-versus-scratch comparison executed successfully end to end

### Requirement: Promotion decision is explicit and non-automatic
The workflow SHALL NOT treat the most recent runtime-compatible glomeruli artifact as promoted unless the promotion report explicitly clears it.

#### Scenario: Candidate clears promotion gates
- **WHEN** exactly one candidate beats the required baselines, avoids degeneracy, passes deterministic prediction review, and satisfies the promotion report criteria
- **THEN** the report marks that candidate as the promoted glomeruli artifact

#### Scenario: Neither candidate clears promotion gates
- **WHEN** both candidates fail the required promotion checks or remain too close to trivial or compatibility baselines
- **THEN** the report blocks promotion
- **AND** no new artifact is labeled as scientifically promoted

#### Scenario: Candidates are scientifically indistinguishable
- **WHEN** both transfer and scratch clear the hard promotion gates but remain within an absolute practical tie margin of `0.02` or less on both shared-manifest Dice and shared-manifest Jaccard
- **THEN** the report records `insufficient_evidence` with an explicit tie reason
- **AND** neither candidate becomes the sole promoted default
- **AND** both artifacts remain available as explicit runtime-compatible research candidates for downstream segmentation and endotheliosis quantification

### Requirement: Scratch candidate training honors the requested crop-size contract
The scratch glomeruli candidate workflow SHALL propagate the caller-supplied crop size through batch-size selection, dynamic cropping, and provenance rather than silently collapsing it to `image_size`.

#### Scenario: Scratch candidate is configured with larger crop context
- **WHEN** scratch training is started with `image_size=256` and `crop_size=512`
- **THEN** the stage-aware batch-size recommendation uses `512` as the crop context for the glomeruli candidate family
- **AND** dynamic patching crops `512`-pixel regions before resizing to the `256`-pixel model input
- **AND** the resulting candidate provenance records both the requested crop size and the output image size

### Requirement: Candidate comparison discloses negative crop supervision coverage
Glomeruli candidate comparison and promotion reports SHALL disclose whether curated negative crop supervision was used to train each candidate and whether deterministic evaluation includes negative-crop coverage.

#### Scenario: Candidate was trained without curated negative crop supervision
- **WHEN** a glomeruli candidate report is generated for a candidate whose training provenance has no supported negative crop manifest
- **THEN** the report records `negative_crop_supervision_status=absent`
- **AND** it does not imply that unmasked MR/TIFF crops were used as true negative supervision

#### Scenario: Candidate was trained with curated negative crop supervision
- **WHEN** a glomeruli candidate report is generated for a candidate whose training provenance includes a supported negative crop manifest
- **THEN** the report records the negative manifest path, manifest hash, negative crop count, source image count, review protocol version, and sampler weight
- **AND** the report states that the supervision is crop-level rather than whole-image negative evidence

#### Scenario: Deterministic promotion evidence includes background crops
- **WHEN** deterministic promotion evidence includes background or negative crop categories
- **THEN** the report distinguishes mask-derived background crops from curated negative crops from unmasked source images
- **AND** it records coverage counts for each source of background evidence separately

### Requirement: Candidate reports disclose negative/background supervision
Glomeruli candidate comparison reports SHALL disclose negative/background supervision state for each candidate.

#### Scenario: Candidate was trained with mask-derived background crops
- **WHEN** a candidate report is generated for a model whose metadata records mask-derived background supervision
- **THEN** `candidate_summary.csv`, `promotion_report.json`, and `promotion_report.md` include mask-derived background crop count, manifest path, manifest hash, and sampler weight

#### Scenario: Candidate was trained with curated MR/TIFF negatives
- **WHEN** a candidate report is generated for a model whose metadata records curated reviewed negative crops
- **THEN** the report includes curated negative crop count, source image count, review protocol version, manifest path, and manifest hash
- **AND** it states that the negatives are crop-level evidence, not whole-image negative evidence

#### Scenario: Candidate lacks negative crop supervision
- **WHEN** a candidate has no validated negative/background supervision metadata
- **THEN** the report records `negative_crop_supervision_status=absent`

### Requirement: Background category gates remain promotion blockers
Promotion gates SHALL continue to block candidates with background false-positive excess even when aggregate Dice is high.

#### Scenario: Aggregate Dice is high but background crops fail
- **WHEN** candidate aggregate Dice clears a nominal threshold
- **AND** deterministic background crops show false-positive foreground excess
- **THEN** the candidate remains not promotion eligible
- **AND** the report lists the background failure reason

### Requirement: Candidate reports disclose augmentation policy
Candidate comparison reports SHALL record the actual augmentation policy used for each candidate.

#### Scenario: Candidate metadata includes augmentation policy
- **WHEN** candidate metadata includes augmentation policy fields
- **THEN** candidate reports include the selected augmentation variant, FastAI transform settings, and whether config-declared augmentation fields were active

### Requirement: Candidate comparison separates runtime use from promotion eligibility
The glomeruli candidate-comparison workflow SHALL distinguish whether an artifact is available for research/runtime use from whether its evidence is eligible for scientific promotion or README-facing current-performance claims.

#### Scenario: Candidate artifact loads but does not clear promotion evidence gates
- **WHEN** a scratch or transfer candidate artifact loads in the supported environment but lacks held-out promotion evidence, split provenance, category support, or prediction-shape evidence
- **THEN** the report SHALL keep the artifact available as `runtime_use_status=available_research_use`
- **AND** it SHALL set `promotion_evidence_status` to `not_promotion_eligible`, `audit_missing`, or `insufficient_evidence_for_promotion`
- **AND** it SHALL NOT describe the artifact as unusable unless model loading or inference itself fails

#### Scenario: Both current candidates are the only available models
- **WHEN** scratch and transfer are the only available glomeruli segmentation candidates
- **THEN** failing promotion gates SHALL NOT remove them from research-use comparison surfaces
- **AND** the report SHALL state that they remain research-use candidates while promotion-facing claims require stronger held-out evidence

### Requirement: Candidate comparison uses held-out-only deterministic promotion manifests
The glomeruli candidate-comparison workflow SHALL build deterministic promotion manifests only from images that are held out from every candidate artifact being evaluated for promotion.

#### Scenario: Candidate comparison trains fresh candidates
- **WHEN** the candidate-comparison workflow trains transfer and no-mitochondria-base candidates
- **THEN** it SHALL persist the shared train/validation split before evaluation
- **AND** it SHALL build the deterministic promotion manifest only from the recorded validation image set
- **AND** it SHALL mark promotion evidence as `not_promotion_eligible` if the deterministic manifest includes any recorded training image

#### Scenario: Candidate comparison evaluates existing artifacts
- **WHEN** `--transfer-model-path` and `--scratch-model-path` are supplied
- **THEN** the workflow SHALL read each artifact's split provenance
- **AND** it SHALL build or validate the deterministic promotion manifest against the intersection of images that are held out for all compared candidate artifacts
- **AND** it SHALL classify any artifact without auditable split provenance as compatibility-only for promotion

#### Scenario: Held-out manifest cannot satisfy category requirements
- **WHEN** the held-out image set cannot provide the required background, boundary, and positive review categories
- **THEN** the workflow SHALL set `promotion_evidence_status=insufficient_evidence_for_promotion`
- **AND** it SHALL use `decision_reason=insufficient_heldout_category_support`
- **AND** candidate artifacts that load successfully SHALL remain `runtime_use_status=available_research_use`
- **AND** no candidate SHALL be marked as promoted or tied for current performance

### Requirement: Candidate comparison reports split-overlap evidence
Candidate comparison reports SHALL expose split-overlap evidence directly in the promotion report artifacts.

#### Scenario: Promotion report is generated
- **WHEN** `promotion_report.json`, `promotion_report.md`, and `promotion_report.html` are written
- **THEN** they SHALL include a split-integrity section with train image overlap count, validation image count, subject overlap count when subject IDs are available, and the exact split provenance source for each candidate
- **AND** any nonzero train overlap SHALL make the evidence `not_promotion_eligible`

#### Scenario: README-facing metrics are requested
- **WHEN** a promotion report has nonzero train overlap, missing split provenance, or audit-failed split integrity
- **THEN** the report SHALL set `promotion_evidence_status=not_promotion_eligible`
- **AND** it SHALL set a machine-readable flag that prevents README-facing current-performance tables from citing its aggregate Dice, Jaccard, precision, or recall as current model performance

### Requirement: Candidate comparison reports transfer-base provenance
Candidate comparison reports SHALL expose the mitochondria base provenance used by transfer candidates.

#### Scenario: Transfer candidate is evaluated
- **WHEN** the transfer candidate is included in candidate comparison
- **THEN** `promotion_report.json`, `promotion_report.md`, and `promotion_report.html` SHALL report the mitochondria base artifact path, `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical training/testing counts, actual fitted image/mask counts, and base resize/preprocessing policy
- **AND** missing transfer-base provenance SHALL set the transfer candidate's `promotion_evidence_status=audit_missing`

#### Scenario: Base used all mitochondria data
- **WHEN** the transfer base reports `mitochondria_training_scope=all_available_pretraining`
- **THEN** candidate comparison MAY evaluate glomeruli transfer against scratch
- **AND** it SHALL NOT cite mitochondria held-out metrics as validation evidence for the base
- **AND** the final glomeruli decision SHALL depend on glomeruli held-out evidence only

#### Scenario: Base claims held-out mitochondria validation
- **WHEN** candidate comparison or documentation describes the base as validated on held-out mitochondria data
- **THEN** the report SHALL require `mitochondria_training_scope=heldout_test_preserved`
- **AND** it SHALL mark the base claim ineligible if the physical mitochondria testing root was included in fitting

### Requirement: Candidate comparison gates foreground-heavy and overcoverage failures
Candidate comparison SHALL include prediction-shape gates that detect broad oversegmentation and excessive foreground burden by review category.

#### Scenario: Candidate predicts foreground on background crops
- **WHEN** a candidate predicts foreground on deterministic background crops above the configured background false-positive limit
- **THEN** the candidate SHALL be marked `promotion_evidence_status=not_promotion_eligible`
- **AND** `prediction_shape_audit.csv` SHALL report the affected examples

#### Scenario: Candidate overcovers positive-like crops
- **WHEN** a candidate's prediction foreground fraction substantially exceeds the truth foreground fraction on positive or boundary crops across the deterministic manifest
- **THEN** the candidate SHALL be marked `promotion_evidence_status=not_promotion_eligible` unless the report records a justified, preconfigured tolerance
- **AND** the report SHALL identify this as overcoverage rather than a high-recall success

#### Scenario: Aggregate metrics hide category failure
- **WHEN** aggregate Dice or Jaccard passes but any required category fails its category-specific gate
- **THEN** the candidate SHALL be marked `promotion_evidence_status=not_promotion_eligible`
- **AND** the final decision SHALL report the failing categories

### Requirement: Candidate comparison reports category-specific metrics
Candidate comparison reports SHALL include per-category and per-cohort/lane metrics in addition to aggregate metrics.

#### Scenario: Candidate metrics are computed
- **WHEN** predictions are evaluated on a deterministic promotion manifest
- **THEN** `metric_by_category.csv` SHALL report metrics by `category`, `cohort_id` when available, `lane_assignment` when available, and candidate family
- **AND** the aggregate report SHALL identify whether performance is driven by foreground-rich positive/boundary crops or by balanced category performance

#### Scenario: Candidate clears only foreground-rich examples
- **WHEN** a candidate performs well on positive and boundary crops but fails background or low-foreground examples
- **THEN** the promotion report SHALL mark the candidate `promotion_evidence_status=not_promotion_eligible`
- **AND** it SHALL NOT summarize the candidate as performing well overall without that limitation

### Requirement: Candidate comparison reports resize-policy sensitivity
Candidate comparison reports SHALL expose whether crop-to-network resizing is helping, hurting, or unproven on held-out deterministic evidence.

#### Scenario: Current resize policy is evaluated
- **WHEN** candidate comparison evaluates a candidate trained with the current `crop_size=512` and `image_size=256` glomeruli policy
- **THEN** the report SHALL include the candidate's resize policy fields and prediction resize-back method
- **AND** `resize_policy_audit.csv` or equivalent structured fields SHALL report source image size, source mask size, crop-to-output resize ratio, held-out metrics, prediction-shape summaries, and threshold/resize ordering by candidate family and review category

#### Scenario: Resolution normalization is imbalanced across splits
- **WHEN** candidate comparison detects materially different source-resolution or resize-ratio distributions between training, validation, and deterministic promotion examples
- **THEN** the report SHALL set `promotion_evidence_status=insufficient_evidence_for_promotion`
- **AND** it SHALL identify the affected cohort ID, lane assignment, split, or review category when available

#### Scenario: Resize sensitivity cannot be run
- **WHEN** no-downsample or less-downsample comparison is infeasible because of memory, runtime, or missing artifact constraints
- **THEN** the report SHALL record `resize_benefit_unproven`
- **AND** promotion-facing documentation SHALL NOT state that the resize policy improves performance

#### Scenario: Resize sensitivity changes the promotion decision
- **WHEN** the resize sensitivity shows that aggregate performance depends on resize-induced foreground inflation, boundary smoothing, or background false positives
- **THEN** the candidate SHALL be marked `promotion_evidence_status=not_promotion_eligible`
- **AND** the report SHALL identify the resize sensitivity as a decision driver

#### Scenario: Threshold and resize ordering changes the decision
- **WHEN** threshold-before-resize versus resize-before-threshold changes category gates, foreground burden gates, or the final candidate decision
- **THEN** the candidate SHALL be marked `promotion_evidence_status=not_promotion_eligible`
- **AND** the report SHALL identify threshold/resize ordering as a decision driver

### Requirement: Candidate comparison reports poor-performance root cause
Candidate comparison reports SHALL explain poor performance with explicit root-cause classes before recommending remediation.

#### Scenario: Candidate performance is poor
- **WHEN** a candidate fails aggregate metrics, category-specific metrics, prediction-shape gates, or visual-panel review
- **THEN** `promotion_report.json`, `promotion_report.md`, and `promotion_report.html` SHALL include the root-cause class, supporting evidence, and next remediation path
- **AND** the report SHALL distinguish code/evaluation defects from true training or supervision limitations

#### Scenario: Retraining is proposed
- **WHEN** the report proposes fresh transfer, scratch, or mitochondria-base training as remediation
- **THEN** it SHALL first show that image/mask pairing, transform alignment, mask binarization, class-channel selection, thresholding, resize policy, split integrity, and panel reproduction checks passed
- **AND** it SHALL identify the remaining root cause as `true_model_underfit`, `mitochondria_base_defect`, `training_signal_insufficient`, or `negative_background_supervision_missing`

### Requirement: Candidate comparison is linked to validation audit artifacts
Candidate comparison SHALL satisfy the pytest-backed validation-audit contract before promotion-facing decisions are accepted.

#### Scenario: Candidate comparison completes
- **WHEN** candidate comparison writes a promotion report
- **THEN** the report SHALL include split-integrity, category, and prediction-shape fields that are covered by `tests/test_segmentation_validation_audit.py`
- **AND** the final promotion evidence status SHALL be `not_promotion_eligible` or `audit_missing` if the pytest-backed validation contract would not be satisfied

#### Scenario: Validation contract is not exercised
- **WHEN** candidate comparison code or report schema changes without corresponding pytest coverage for split integrity, category metrics, prediction-shape gates, and documentation claim eligibility
- **THEN** the change SHALL be incomplete
- **AND** promotion-facing documentation claims SHALL NOT be refreshed from that report until the pytest contract passes

### Requirement: Candidate comparison uses a canonical workflow-config control surface
The repository SHALL treat a dedicated candidate-comparison workflow config run through `eq run-config` as the authoritative top-level control surface for glomeruli promotion provenance. The underlying training and comparison commands SHALL remain worker surfaces whose exact invocations are recorded in the workflow provenance, but they SHALL NOT compete as separate orchestration contracts.

#### Scenario: Candidate comparison is configured
- **WHEN** transfer and no-mitochondria-base candidate runs are defined for promotion comparison
- **THEN** the authoritative recipe is expressed through the dedicated workflow config `configs/glomeruli_candidate_comparison.yaml` with `workflow: glomeruli_candidate_comparison`
- **AND** the workflow provenance records the exact underlying training and comparison commands that were launched

#### Scenario: Candidate comparison output location is not supplied
- **WHEN** the candidate-comparison workflow is executed without an explicit output override
- **THEN** it SHALL write promotion reports, deterministic manifests, metrics, and review assets under the active runtime output root's `output/segmentation_evaluation/glomeruli_candidate_comparison/` subtree
- **AND** it MAY still accept an explicit caller-supplied override path when the user intentionally wants a different destination

#### Scenario: Candidate comparison trains model artifacts
- **WHEN** the candidate-comparison workflow trains transfer or no-base candidates
- **THEN** trained candidate model artifacts SHALL be written under the configured model root's glomeruli segmentation subtrees
- **AND** the comparison output tree SHALL reference those artifacts rather than duplicating them under the evaluation report directory

