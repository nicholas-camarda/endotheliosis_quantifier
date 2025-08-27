# JavaScript Code Style Guide

## Core Principles

### 1. **Code Organization**
- Use descriptive function and variable names
- Keep functions focused and reasonably sized
- Use consistent formatting and indentation
- Document complex logic with comments

### 2. **Module Management**
- Use ES6 modules (`import`/`export`)
- Prefer named exports over default exports
- Keep dependencies minimal and well-documented
- Use package managers (npm/yarn) for external libraries

### 3. **File Structure**
- Put code in `scripts/` directory
- Put tests in `tests/` directory
- Use descriptive file names
- Organize related functions together

### 4. **Documentation**
- Use JSDoc for function documentation
- Include parameter descriptions and return values
- Add examples for complex functions
- Keep documentation simple and clear

## Testing

### Basic Test Structure
```javascript
// tests/functionName.test.js
const { yourFunction } = require('../scripts/yourModule');

test('function does what it should', () => {
  // Setup test data
  const testData = [1, 2, 3, 4, 5];
  
  // Test the function
  const result = yourFunction(testData);
  
  // Check the result
  expect(result).toHaveLength(5);
  expect(result.every(x => x > 0)).toBe(true);
});
```

### Running Tests
```bash
# Run all tests
npm test

# Run specific test file
npm test tests/functionName.test.js

# Run with verbose output
npm test -- --verbose
```

### Test Best Practices
- Test actual behavior, not just existence
- Use realistic test data
- Test both success and error cases
- Keep tests simple and focused

## Code Style

### Formatting
- Use 2 spaces for indentation
- Use semicolons consistently
- Use descriptive variable names
- Follow consistent naming conventions

### Comments
- Comment complex logic
- Explain why, not what
- Keep comments up to date
- Use clear, concise language

## Best Practices

- **Simplicity**: Prefer simple, clear solutions
- **Consistency**: Use consistent patterns throughout
- **Readability**: Write code that's easy to understand
- **Maintainability**: Structure code for easy modification
