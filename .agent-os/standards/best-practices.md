# Best Practices

## Overview

This directory contains simplified development standards focused on core principles rather than extensive rules. These standards are designed to be clear, practical, flexible, and minimal.

## Core Documents

### [Methodology](./methodology.md)
Core development principles and testing guidelines.

### [Testing Standards](./testing-standards.md)
Simple testing principles and language-specific examples.

### [R Style](./code-style/r-style.md)
R coding standards and testing examples.

### [Python Style](./code-style/python-style.md)
Python coding standards and testing examples.

### [JavaScript Style](./code-style/javascript-style.md)
JavaScript coding standards and testing examples.

## Core Principles

### 1. **Code Quality**
- Write readable, maintainable code
- Use descriptive names for functions, variables, and files
- Keep functions focused and reasonably sized
- Document complex logic with clear comments

### 2. **Testing**
- Test actual behavior, not just existence
- Use realistic test data
- Keep tests simple and focused
- Test both success and error cases when relevant

### 3. **Documentation**
- Document the purpose and usage of functions
- Keep documentation simple and focused
- Update documentation when code changes
- Use examples for complex functionality

### 4. **Organization**
- Organize code logically
- Use consistent file and directory structures
- Group related functionality together
- Keep dependencies minimal

### 5. **Problem Solving**
- Understand the problem before writing code
- Choose the simplest solution that works
- Focus on effectiveness over complexity
- Test your assumptions

## Runtime Verification (THE ONLY TESTING APPROACH)

**The ONLY way to test your code is to run the main execution script and check output logs and errors from main runs on real data.**

### **What This Means:**
- **Run your actual analysis scripts** with real data
- **Check the output logs** to see if the analysis worked
- **Look for errors** in the logs to see what broke
- **Verify the results** match what you expected

### **The Process:**
1. **Run the main execution script** (your actual analysis)
2. **Capture all output** to log files
3. **Check the logs** for success/error messages
4. **Verify the results** are what you expected
5. **If it failed**, fix the code and run again

### **Example:**
```bash
# Run your actual analysis
Rscript scripts/main_analysis.R > test_output/analysis.log 2>&1

# Check if it worked
grep "ERROR" test_output/analysis.log
grep "Analysis completed successfully" test_output/analysis.log

# Check the actual output files
ls -la test_output/results/
```

### **What NOT to Do:**
- ❌ **Don't write testthat unit tests** - they're not useful for your workflow
- ❌ **Don't use Rscript for "validation"** - that's not testing
- ❌ **Don't create fake test data** - use your real data
- ❌ **Don't write formal test frameworks** - just run your analysis

**REMEMBER: Testing = Running your actual analysis and checking the logs**

## Key Principles

- **Simplicity**: Prefer simple, clear solutions over complex ones
- **Focus**: Work on one thing at a time
- **Effectiveness**: Choose the approach that gets the job done well
- **Testing**: Test what matters with realistic data
- **Documentation**: Keep it simple and useful

## AI Tool Usage

### **Context7 for Documentation**
- Use Context7 to get up-to-date documentation for libraries and frameworks
- Use Context7 when implementing new patterns or approaches
- Use Context7 to verify current best practices
- Use Context7 for language-specific examples and syntax

### **Sequential-Thinking for Planning**
- Use sequential-thinking when planning development strategies
- Use sequential-thinking to break down complex problems
- Use sequential-thinking to identify potential issues and risks
- Use sequential-thinking to plan implementation approaches

### **When to Use These Tools**
- Before starting any development task
- When choosing between implementation approaches
- When debugging complex issues
- When implementing new features or patterns
- When working with unfamiliar libraries or frameworks
- When planning testing strategies

## Development Workflow

### Before Writing Code
- Understand the requirements clearly
- Plan your approach
- Consider existing solutions
- Think about testing

### While Writing Code
- Write clear, readable code
- Test as you go
- Keep functions focused
- Use appropriate tools and libraries

### After Writing Code
- Test your implementation
- Review and refactor if needed
- Update documentation
- Consider edge cases

### Runtime Verification Loop (Strongly Recommended)
- Run the smallest reproducible slice to produce output/logs
- Persist logs to `test_output/` or `logs/`
- Verify logs/output with exact content checks (strings/regex/structured checks)
- Improve logging to capture FULL errors (no truncation)
- Re-run and re-check until verification passes

## Philosophy

These standards are designed to be:
- **Clear**: Easy to understand and follow
- **Practical**: Focused on real-world effectiveness
- **Flexible**: Adaptable to different situations
- **Minimal**: Only what's necessary for quality code

## Best Practices Summary

- **Simplicity**: Prefer simple, clear solutions over complex ones
- **Focus**: Work on one thing at a time
- **Effectiveness**: Choose the approach that gets the job done well
- **Maintainability**: Write code that's easy to understand and modify
- **Testing**: Test what matters, not everything
- **Documentation**: Keep it simple and useful
