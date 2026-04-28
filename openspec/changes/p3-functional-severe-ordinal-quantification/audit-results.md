# P3 Audit Results

This file will be filled during `/apply`.

Initial planning evidence from P2:

- P2 selected severe-aware estimator missed most `score >= 2` images.
- Reviewer adjudication removed some over-graded severe false negatives but left many true severe misses.
- Manual review did not identify recurrent glomerulus/non-glomerulus mask failure as the primary bottleneck.
- A read-only screen found recoverable severe signal in existing morphology features when using class-balanced severe-risk modeling: morphology-only balanced logistic AUROC `0.829`, average precision `0.326`, recall `0.676` / precision `0.262` / false negatives `23/71` at threshold `0.5`, recall `0.803` / precision `0.288` / false negatives `14/71` near threshold `0.477`, and recall `0.944` / precision `0.211` / false negatives `4/71` at threshold `0.3`.

Initial planning decision:

- P3 should attempt a final current-data product, not only another diagnostic report.
- The first product target is high-sensitivity severe-risk triage.
- Ordinal burden is not declared impossible. P3 must test three-band, four-band, and six-bin ordinal outputs before downgrading ordinal output to diagnostic-only or unsupported.
- Learned ROI and embedding-heavy neural features are not excluded. P3 must test them as explicit gated severe and ordinal candidate lanes; prior artifacts prove overfit/source-sensitivity risk in older objectives, not impossibility for the P3 target.
- Scalar burden and external validation remain non-reportable unless gates unexpectedly pass.
