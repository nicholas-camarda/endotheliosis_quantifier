# Tests Directory

This directory contains organized test files for the endotheliosis quantifier project.

## Directory Structure

```
tests/
├── unit/                    # Unit tests for individual functions/modules
├── integration/             # Integration tests for pipelines and workflows  
├── evaluation/             # Model evaluation and performance testing
└── README.md               # This file
```

## Test Categories

### Unit Tests (`unit/`)
Tests for individual functions, classes, and modules:
- Data loading functions
- Model components
- Utility functions
- Configuration management
- Feature extraction

### Integration Tests (`integration/`)
Tests for complete workflows and pipelines:
- CLI interface testing
- Pipeline integration
- Multi-stage workflows
- Infrastructure testing

### Evaluation Tests (`evaluation/`)
Model evaluation and performance testing:
- Model accuracy evaluation
- Performance benchmarking
- Comparison testing

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific category
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/evaluation/

# Run specific test file
python -m pytest tests/unit/test_specific_file.py
```

## Cleanup History

This directory was cleaned up from 73 files to 44 organized files:
- **Removed**: 17 debug/temporary files
- **Archived**: 5 analysis files (moved to archive/old_tests/)
- **Consolidated**: 7 evaluation files → 3 essential files
- **Organized**: Remaining 42 test files into logical categories

## Archive

Old test files have been moved to `../archive/old_tests/` with subdirectories:
- `analysis/` - Historical analysis and debugging files
- `evaluation/` - Old evaluation file versions
