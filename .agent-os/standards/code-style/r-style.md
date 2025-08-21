# R Code Style Guide

## Core Principles

### 1. **Code Organization**
- Use descriptive function names
- Keep functions focused and reasonably sized
- Use consistent formatting and indentation
- Document complex logic with comments

### 2. **Package Management**
- Use `library()` to load packages
- Prefer tidyverse for data manipulation
- Use base R when it provides clear advantages
- Keep package dependencies minimal

### 3. **File Structure**
- Organize your code in a way that makes sense for your project
- Put analysis scripts in `scripts/` directory
- Use descriptive file names
- Group related functions together

### 4. **Documentation**
- Use roxygen2 for function documentation
- Include parameter descriptions and return values
- Add examples for complex functions
- Keep documentation simple and clear

## Runtime Verification (THE ONLY TESTING APPROACH)

### Project Structure
```
your_project/
├── .Rproj
├── scripts/
│   ├── load_all.R    # Master file: loads all functions AND packages
│   ├── main_analysis.R         # Main execution script
│   ├── utils/
│   │   ├── data_helpers.R
│   │   └── plotting_functions.R
│   ├── models/
│   │   ├── regression.R
│   │   └── classification.R
│   └── data/
│       └── preprocessing.R
├── test_output/      # All logs and results go here
│   ├── analysis.log
│   └── results/
└── other_files
```

### Master Environment File
Your `load_all.R` file loads everything needed:

```r
# scripts/load_all.R
# Load all required packages
library(tidyverse)
library(here)
# ... other packages

# Load all utility functions
source(here("utils", "data_helpers.R"))
source(here("utils", "plotting_functions.R"))

# Load model functions
source(here("models", "regression.R"))
source(here("models", "classification.R"))

# Load data processing functions
source(here("data", "preprocessing.R"))
```

## Logging Requirements

### Your Analysis Scripts Must Log Everything
```r
# scripts/main_analysis.R
cat("Starting GEP analysis...\n")
cat("Loading data from:", data_file, "\n")

# Load everything
source(here("scripts", "load_all.R"))

# After each major step
cat("Data loaded successfully. Rows:", nrow(data), "\n")
cat("Analysis step 1 completed\n")

# When finished
cat("GEP analysis completed successfully\n")
cat("Results saved to:", output_file, "\n")
```

## Running and Verifying

### Run Your Analysis
```bash
# Run the actual analysis
Rscript scripts/main_analysis.R > test_output/analysis.log 2>&1
```

### Check if It Worked
```bash
# Look for success message
grep "completed successfully" test_output/analysis.log

# Look for errors
grep "ERROR\|Error\|error" test_output/analysis.log

# Check if output files were created
ls -la test_output/results/
```

## What NOT to Do

### ❌ NEVER DO THESE:
- **Don't write testthat unit tests** - they're not useful for your workflow
- **Don't use Rscript for "validation"** - that's not testing
- **Don't create fake test data** - use your real data
- **Don't write formal test frameworks** - just run your analysis

### ❌ WRONG APPROACH (NEVER DO THIS):
```r
# This is NOT testing - it's just checking if files exist
Rscript -e "library(readxl); file <- 'test_output/tmp/04_GEP_Validation/unified_summary/full_cohort_simple_gep_validation.xlsx'; if(file.exists(file)) { print('File exists') }"
```

### ✅ RIGHT APPROACH (ALWAYS DO THIS):
```bash
# Run the actual analysis
Rscript scripts/gep_analysis.R > test_output/gep_run.log 2>&1

# Check if it actually worked
grep "ERROR" test_output/gep_run.log
grep "Analysis completed" test_output/gep_run.log

# Check the actual results
ls -la test_output/results/
```

## Summary

**Testing = Running your actual analysis and checking the logs**

- **Run**: Your main analysis script
- **Check**: Output logs for success/errors  
- **Verify**: Results files were created
- **Repeat**: If it failed, fix and run again

**That's it. No testthat. No unit tests. No fake data. Just run your analysis and check if it worked.**
