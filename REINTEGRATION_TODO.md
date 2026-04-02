# Reintegration TODO

## Current Status

- [x] Create reintegration branch from stabilized baseline
- [x] Commit baseline stabilization and docs cleanup
- [x] Import initial smoke tests and Lucchi organizer utility
- [x] Add hardware detection coverage
- [x] Add `organize-lucchi` CLI command
- [x] Review `backup/master-20250914-0900` for weight-loading improvements only

## Donor Review

- [x] Review safe config and data-management helpers from `origin/model-retraining-infrastructure`
- [x] Review `backup/master-20250914-0900` for weight-loading improvements only

## Integration Work

- [ ] Import selected low-risk path/config improvements
- [ ] Import any safe backup-branch training improvements manually
- [ ] Re-run focused validation after each integration slice

## Active Slice

- [x] Add Jaccard metric tracking to current training entrypoints
- [ ] Normalize repo-aware path helpers and config defaults
- [ ] Clean up output manager summary formatting and path resolution
- [ ] Add unit coverage for path/config/output manager behavior

## Merge Preparation

- [ ] Reassess branch cleanliness and test coverage on `codex/reintegrate-recent-work`
- [ ] Decide whether the branch is ready to merge back into `master`
- [ ] Push branch and prepare merge sequence once stable
