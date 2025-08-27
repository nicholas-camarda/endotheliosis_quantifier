
# Testing Standards

## Core Purpose

**The ONLY way to test your code is to run the main execution script and check output logs and errors from main runs on real data.**

## Runtime Verification Loop (THE ONLY TESTING APPROACH)

### What This Means
- **Run your actual analysis scripts** with real data
- **Check the output logs** to see if the analysis worked
- **Look for errors** in the logs to see what broke
- **Verify the results** match what you expected

### The Process
1. **Run the main execution script** (your actual analysis)
2. **Capture all output** to log files
3. **Check the logs** for success/error messages
4. **Verify the results** are what you expected
5. **If it failed**, fix the code and run again

### Example
```bash
# Run your actual analysis
Rscript scripts/main_analysis.R > test_output/analysis.log 2>&1

# Check if it worked
grep "ERROR" test_output/analysis.log
grep "Analysis completed successfully" test_output/analysis.log

# Check the actual output files
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

## Logging Requirements

### Your Code Must Log Everything
- **Log all major steps** in your analysis
- **Log any errors** with full error messages
- **Log completion status** when analysis finishes
- **Log file paths** where results are saved

### Example Logging in Your R Code
```r
# In your analysis script
cat("Starting GEP analysis...\n")
cat("Loading data from:", data_file, "\n")

# After each major step
cat("Data loaded successfully. Rows:", nrow(data), "\n")
cat("Analysis step 1 completed\n")

# When finished
cat("GEP analysis completed successfully\n")
cat("Results saved to:", output_file, "\n")
```

## Verification Commands

### Check if Analysis Worked
```bash
# Look for success message
grep "completed successfully" test_output/analysis.log

# Look for errors
grep "ERROR\|Error\|error" test_output/analysis.log

# Check if output files were created
ls -la test_output/results/
```

### Check Specific Results
```bash
# Check if specific files exist
ls test_output/results/gep_validation.xlsx

# Check file sizes (should be > 0)
ls -la test_output/results/*.xlsx
```

## Summary

**Testing = Running your actual analysis and checking the logs**

- **Run**: Your main analysis script
- **Check**: Output logs for success/errors  
- **Verify**: Results files were created
- **Repeat**: If it failed, fix and run again

**That's it. No testthat. No unit tests. No fake data. Just run your analysis and check if it worked.** 