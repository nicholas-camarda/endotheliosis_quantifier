## ADDED Requirements

### Requirement: Segmentation-backed embeddings use training-equivalent preprocessing
Quantification embeddings derived from segmentation backbones SHALL use the same ImageNet-normalized preprocessing contract used for segmentation training.

#### Scenario: Embeddings are generated
- **WHEN** the quantification workflow generates segmentation-backbone embeddings
- **THEN** each image tensor SHALL be preprocessed through the shared segmentation preprocessing function
- **AND** `embedding_summary.json` SHALL record `inference_preprocessing="imagenet_normalized"`

#### Scenario: Preprocessing changes
- **WHEN** the shared segmentation preprocessing function changes
- **THEN** embedding and GPU inference tests SHALL verify equivalence to the training DataBlock normalization contract

### Requirement: ROI extraction fails closed on invalid geometry
Quantification ROI extraction SHALL require image/mask shape agreement and SHALL NOT write ROI crops for masks with no component passing the minimum-area gate.

#### Scenario: Image and mask dimensions differ
- **WHEN** a quantification row has an image and mask with different height or width
- **THEN** ROI extraction SHALL fail or record `roi_status="image_mask_size_mismatch"`
- **AND** no ROI crop SHALL be written for that row

#### Scenario: All components are below minimum area
- **WHEN** every positive mask component is smaller than the configured minimum component area
- **THEN** ROI extraction SHALL record `roi_status="component_below_min_area"`
- **AND** it SHALL NOT fall back to the entire positive mask
- **AND** no ROI crop SHALL be written for that row

### Requirement: Inference threshold is explicit and recorded
Segmentation inference used by quantification SHALL use an explicit threshold or the core default threshold and SHALL record that threshold in provenance.

#### Scenario: Batch inference runs without threshold argument
- **WHEN** batch GPU inference is called without an explicit threshold
- **THEN** it SHALL use `DEFAULT_PREDICTION_THRESHOLD`
- **AND** output provenance SHALL record the threshold value and source

#### Scenario: Batch inference runs with threshold argument
- **WHEN** batch GPU inference is called with an explicit threshold
- **THEN** it SHALL use that threshold
- **AND** output provenance SHALL record the threshold value and source
