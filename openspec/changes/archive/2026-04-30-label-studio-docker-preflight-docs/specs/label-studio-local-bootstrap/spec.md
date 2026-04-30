## MODIFIED Requirements

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
