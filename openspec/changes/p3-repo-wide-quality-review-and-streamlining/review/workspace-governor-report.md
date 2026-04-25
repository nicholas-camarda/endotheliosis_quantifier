# Workspace Governor Report

## Scope

- Lane: workspace-governor non-mutating assessment
- Repo: `/Users/ncamarda/Projects/endotheliosis_quantifier`
- Status: `ok`
- Generated: `20260425T024326Z`

## Direct Evidence

- Current root kind: `projects`
- Profile guess: `sideproject` confidence `0.9`
- Proposed code root: `/Users/ncamarda/Projects/endotheliosis_quantifier`
- Proposed runtime root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`
- Proposed cloud home: `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier`
- Publish root name: `Analysis` from `analysis_registry`
- Public/private doc contract passed: `True`
- Rewrite candidate count: `0`
- Unresolved question count: `1`
  - Which test or smoke command should define migration success?

## Classification

- `/Users/ncamarda/Projects/endotheliosis_quantifier` action `keep` reason `already compliant` destination `None`

## Findings

- No public-doc rewrite candidates reported by workspace-governor.
- Migration success smoke command remains the only unresolved workspace-governor question; p3 resolves this with OpenSpec validation, config dry-runs, full pytest, and a shortened 5-epoch candidate-comparison run.

## Direct Evidence vs Inference

- Direct evidence: `analysis_registry.yaml`, `AGENTS.md`, repo location, workspace-governor dry-run/audit/publish-preview inspection.
- Inference: no move is needed because the repo is already in the expected code root and registry declares runtime/cloud roots.

## Recommended Actions

- Keep source/runtime/cloud roots separate.
- Do not mutate workspace layout in p3; no move candidates are accepted.
- Use final validation and the 5-epoch quick run as the migration-success smoke evidence requested by workspace-governor.
