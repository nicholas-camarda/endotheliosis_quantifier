# Spec Tasks

## Tasks

- [ ] 1. Standardize Conda environment to `eq`
  - [ ] 1.1 Write tests for environment: verify `environment.yml` has `name: eq`; confirm `conda env list` includes `eq`
  - [ ] 1.2 Update `environment.yml` to `name: eq` if needed
  - [ ] 1.3 Update docs to reference `conda activate eq` (README, product docs)
  - [ ] 1.4 Verify: `conda env create -f environment.yml` or `conda env update -f environment.yml --prune`; run `scripts/utils/runtime_check.py` successfully

- [ ] 2. Implement repository rename to `endotheliosis_quantifier`
  - [x] 2.1 Write tests for remotes: capture `git remote -v`; verify or note required URL change
  - [x] 2.2 Document GitHub rename impacts and local steps (rename folder, update remote): add Migration section to README
  - [x] 2.3 Update references in docs from `endotheliosisQuantifier_LEARN` to `endotheliosis_quantifier`
  - [x] 2.4 Verify: clone via new URL; run quick commands to ensure setup works

- [ ] 3. Update product documentation with migration notes
  - [ ] 3.1 Write tests for docs: grep for old repo name in docs; ensure no stale references remain
  - [x] 3.2 Add migration notes link in `.agent-os/product/README.md`
  - [ ] 3.3 Verify links resolve and commands are correct across macOS/WSL2
