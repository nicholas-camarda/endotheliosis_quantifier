# label-studio-local-bootstrap Specification

## Purpose

Define the one-command local bootstrap workflow that lets an admin point `eq` at a directory of images and open a ready-to-grade local Label Studio glomerulus-grading project.

## Requirements

### Requirement: CLI SHALL bootstrap Label Studio from an image directory

The system SHALL provide `eq labelstudio start --images <image-dir>` to prepare a local Label Studio glomerulus-grading project from a user-provided image directory.

#### Scenario: Image directory is valid

- **WHEN** the user runs `eq labelstudio start --images /path/to/images`
- **THEN** the command discovers supported images recursively
- **AND** prepares a Label Studio project using `configs/label_studio_glomerulus_grading.xml`
- **AND** prints the Label Studio URL and project URL

#### Scenario: Image directory is missing

- **WHEN** the user runs `eq labelstudio start --images /missing/path`
- **THEN** the command fails before starting Docker or calling Label Studio
- **AND** the error identifies the missing image directory

### Requirement: Bootstrap SHALL preserve image metadata for downstream grading

The system SHALL preserve each imported image's relative path metadata so later exports can be traced back to source files.

#### Scenario: Nested image directory is imported

- **WHEN** the image root contains `animal_1/kidney_a/image_001.tif`
- **THEN** the generated task includes `source_relative_path = animal_1/kidney_a/image_001.tif`
- **AND** includes `source_filename = image_001.tif`
- **AND** includes `subject_hint = animal_1`

#### Scenario: Flat image directory is imported

- **WHEN** the image root contains `image_001.tif`
- **THEN** dry-run task generation includes `subject_hint = image_001`
- **AND** full Label Studio API bootstrap fails with an actionable error requiring images to be placed in at least one subfolder under `--images`

### Requirement: Runtime artifacts SHALL stay outside Git

The system SHALL write Label Studio runtime files under the active runtime root by default and SHALL NOT create raw images, databases, imports, or exports under Git-tracked repo data roots.

#### Scenario: Default runtime root is used

- **WHEN** no `--runtime-root` is provided
- **THEN** the command uses `get_active_runtime_root() / "labelstudio"` for Label Studio runtime artifacts

#### Scenario: Dry run is requested

- **WHEN** the user passes `--dry-run`
- **THEN** the command prints the planned runtime paths, Docker container name, project title, and image count
- **AND** does not start Docker or call the Label Studio API

### Requirement: Bootstrap SHALL use local Docker Label Studio without polluting eq-mac

The system SHALL run Label Studio as a separate Docker service and SHALL NOT install Label Studio into the project Python environment.

#### Scenario: Docker is unavailable

- **WHEN** Docker cannot be found or cannot run and the command is not in dry-run mode
- **THEN** the command fails with an actionable Docker availability error before attempting project import

#### Scenario: Docker Desktop is installed but stopped on macOS

- **WHEN** Docker Desktop is installed at `/Applications/Docker.app` but `docker info` reports that the daemon is unavailable
- **THEN** the command attempts to start Docker Desktop with `open -a Docker`
- **AND** waits for Docker to become available before attempting project import

#### Scenario: Docker Desktop is missing on macOS

- **WHEN** Docker is not installed on macOS
- **THEN** the command fails with the install command `brew install --cask docker`

#### Scenario: Docker command is planned

- **WHEN** the command prepares Label Studio startup
- **THEN** the Docker invocation includes the configured port, runtime data mount, read-only image media mount, local file serving environment variables, username, password, and API token

#### Scenario: Login credentials are reported

- **WHEN** the command starts or dry-runs a local Label Studio project
- **THEN** the command output includes the configured login email and password

#### Scenario: Local image files are served to the labeling UI

- **WHEN** the command imports image tasks from a nested image directory
- **THEN** the command creates Local Files import storage entries for the project before task import
- **AND** each imported task image URL resolves through `/data/local-files/?d=...`

### Requirement: Bootstrap SHALL configure and import through Label Studio API

The system SHALL create or reuse a Label Studio project, apply the glomerulus grading XML config, and import generated image tasks through the Label Studio HTTP API.

#### Scenario: API bootstrap succeeds

- **WHEN** Label Studio is reachable at the configured URL
- **THEN** the command creates or reuses the project titled `EQ Glomerulus Grading`
- **AND** imports the generated image tasks into that project

#### Scenario: API bootstrap fails

- **WHEN** Label Studio does not become reachable within the startup timeout
- **THEN** the command fails with the URL, timeout, and next diagnostic command to inspect the Docker container