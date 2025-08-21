---
description: Rules to execute a task and its sub-tasks using Agent OS
globs:
alwaysApply: false
version: 1.0
encoding: UTF-8
---

# Task Execution Rules

## Overview

Execute a specific task along with its sub-tasks systematically following a TDD development workflow.

<pre_flight_check>
  EXECUTE: @~/.agent-os/instructions/meta/pre-flight.md
  NOTE: Path conventions — Home: @~/.agent-os, Project: @.agent-os
  PROJECT_CONTEXT_CHECK:
    **MANDATORY**: ALWAYS read @.agent-os/project-context.md first (current workspace)
    VERIFY: Dataset names, function names, file paths are correct
    UPDATE: Context file if new key information is learned
    REFERENCE: This file when needing project details
  WEB_SEARCH_CHECK:
    IF uncertain OR current info required:
      USE: @Web to fetch latest official docs/best practices (never curl)
  MCP_CONTEXT7_CHECK:
    WHEN: docs/api/library/examples/usage are needed OR library usage/version is uncertain
    USE: Context7 (resolve-library-id → get-library-docs) to fetch authoritative docs
  SEQUENTIAL_THINKING:
    ACTION: Use sequential-thinking MCP to outline steps, dependencies, and minimum-size test plan before coding
  STANDARDS_INDEX_CHECK: If unsure which standard to consult, READ @~/.agent-os/standards/README.md to pick targets
</pre_flight_check>


<process_flow>

<step number="1" name="task_understanding">

### Step 1: Task Understanding

Read and analyze the given parent task and all its sub-tasks from tasks.md to gain complete understanding of what needs to be built.

<task_analysis>
  <read_from_tasks_md>
    - Parent task description
    - All sub-task descriptions
    - Task dependencies
    - Expected outcomes
  </read_from_tasks_md>
</task_analysis>

<instructions>
  ACTION: Read the specific parent task and all its sub-tasks
  ANALYZE: Full scope of implementation required
  UNDERSTAND: Dependencies and expected deliverables
  NOTE: Test requirements for each sub-task
</instructions>

</step>

<step number="2" name="technical_spec_review">

### Step 2: Technical Specification Review

Search and extract relevant sections from technical-spec.md to understand the technical implementation approach for this task.

<selective_reading>
  <search_technical_spec>
    FIND sections in technical-spec.md related to:
    - Current task functionality
    - Implementation approach for this feature
    - Integration requirements
    - Performance criteria
  </search_technical_spec>
</selective_reading>

<instructions>
  ACTION: Search technical-spec.md for task-relevant sections
  EXTRACT: Only implementation details for current task
  SKIP: Unrelated technical specifications
  FOCUS: Technical approach for this specific feature
</instructions>

</step>

<step number="3" name="existing_functionality_verification">

### Step 3: Existing Functionality Verification

**CRITICAL: Verify existing functionality for the current task before attempting implementation**

Before implementing new features or modifications, thoroughly check what already exists to avoid duplication and understand integration points.

<verification_requirements>
  <codebase_analysis>
    - Search for existing functions/methods that might handle similar functionality
    - Check for existing UI components or patterns that could be reused
    - Identify existing database models or schemas that might be relevant
    - Review existing API endpoints that could be extended
  </codebase_analysis>
  <integration_check>
    - Verify how new functionality will integrate with existing systems
    - Check for existing patterns or conventions to follow
    - Identify potential conflicts with current implementation
    - Assess impact on existing functionality
  </integration_check>
  <duplication_prevention>
    - Ensure new implementation doesn't duplicate existing code
    - Check if existing functionality can be extended instead of replaced
    - Verify that new features build upon existing architecture
  </duplication_prevention>
</verification_requirements>

<verification_process>
  <search_existing_code>
    1. Search codebase for similar function names or patterns
    2. Check existing test files for related functionality
    3. Review documentation or comments for existing features
    4. Examine database schema for relevant tables/columns
  </search_existing_code>
  <analyze_existing_implementation>
    1. Understand how existing functionality works
    2. Identify patterns and conventions used
    3. Check for any technical debt or issues
    4. Assess quality and maintainability of existing code
  </analyze_existing_implementation>
  <plan_integration>
    1. Determine if new functionality should extend existing code
    2. Plan how to integrate with current patterns
    3. Identify any refactoring needed for existing code
    4. Ensure backward compatibility is maintained
  </plan_integration>
</verification_process>

<instructions>
  **ACTION**: Thoroughly search and analyze existing functionality before implementation
  **PURPOSE**: Prevent duplication, understand integration points, maintain consistency
  **REQUIREMENT**: No implementation begins without understanding existing codebase
  **OUTPUT**: Clear understanding of what exists and how to integrate new functionality
</instructions>

</step>

<step number="4" subagent="context-fetcher" name="best_practices_review">

### Step 4: Best Practices Review

Use the context-fetcher subagent to retrieve relevant sections from @~/.agent-os/standards/best-practices.md that apply to the current task's technology stack and feature type.

<selective_reading>
  <search_best_practices>
    FIND sections relevant to:
    - Task's technology stack
    - Feature type being implemented
    - Testing approaches needed
    - Code organization patterns
  </search_best_practices>
</selective_reading>

<instructions>
  ACTION: Use context-fetcher subagent
  REQUEST: "Find best practices sections relevant to:
            - Task's technology stack: [CURRENT_TECH]
            - Feature type: [CURRENT_FEATURE_TYPE]
            - Testing approaches needed
            - Code organization patterns"
  PROCESS: Returned best practices
  APPLY: Relevant patterns to implementation
</instructions>

</step>

<step number="5" subagent="context-fetcher" name="testing_standards_review">

### Step 5: Testing Standards Review

Use the context-fetcher subagent to retrieve relevant testing standards from @~/.agent-os/standards/testing-standards.md for the languages and file types being used in this task.

<selective_reading>
  <search_testing_standards>
    FIND testing standards for:
    - Languages used in this task
    - File types being modified
    - Component patterns being implemented
    - Testing style guidelines
  </search_testing_standards>
</selective_reading>

<instructions>
  ACTION: Use context-fetcher subagent
  REQUEST: "Find testing standards for:
            - Languages: [LANGUAGES_IN_TASK]
            - File types: [FILE_TYPES_BEING_MODIFIED]
            - Component patterns: [PATTERNS_BEING_IMPLEMENTED]
            - Testing style guidelines"
  PROCESS: Returned style rules
  APPLY: Relevant formatting and patterns
</instructions>

</step>

<step number="5.1" subagent="context-fetcher" name="code_style_review">

### Step 5.1: Code Style Review (Language-Specific)

Use the context-fetcher subagent to retrieve relevant language-specific code style rules from @~/.agent-os/standards/code-style/[language]-style.md as needed.

<selective_reading>
  <search_code_style>
    FIND code style rules for:
    - Languages used in this task
    - File types being modified
  </search_code_style>
</selective_reading>

<instructions>
  ACTION: Use context-fetcher subagent
  REQUEST: "Find code style rules for:
            - Languages: [LANGUAGES_IN_TASK]
            - File types: [FILE_TYPES_BEING_MODIFIED]"
  PROCESS: Returned style rules
  APPLY: Relevant formatting and patterns
</instructions>

</step>

<step number="6" name="task_execution">

### Step 6: Task and Sub-task Execution

Execute the parent task and all sub-tasks in order using test-driven development (TDD) approach.

<typical_task_structure>
  <first_subtask>Write tests for [feature]</first_subtask>
  <middle_subtasks>Implementation steps</middle_subtasks>
  <final_subtask>Verify all tests pass</final_subtask>
</typical_task_structure>

<execution_order>
  <subtask_1_tests>
    IF sub-task 1 is "Write tests for [feature]":
      - Write all tests for the parent feature
      - Include unit tests, integration tests, edge cases
      - Run tests to ensure they fail appropriately
      - Mark sub-task 1 complete
  </subtask_1_tests>

  <middle_subtasks_implementation>
    FOR each implementation sub-task (2 through n-1):
      - Implement the specific functionality
      - Make relevant tests pass
      - Update any adjacent/related tests if needed
      - Refactor while keeping tests green
      - Mark sub-task complete
  </middle_subtasks_implementation>

  <final_subtask_verification>
    IF final sub-task is "Verify all tests pass":
      - Run entire test suite
      - Fix any remaining failures
      - Ensure no regressions
      - Mark final sub-task complete
  </final_subtask_verification>
</execution_order>

<test_management>
  <new_tests>
    - Written in first sub-task
    - Cover all aspects of parent feature
    - Include edge cases and error handling
  </new_tests>
  <test_updates>
    - Made during implementation sub-tasks
    - Update expectations for changed behavior
    - Maintain backward compatibility
  </test_updates>
</test_management>

<instructions>
  ACTION: Execute sub-tasks in their defined order
  RECOGNIZE: First sub-task typically writes all tests
  IMPLEMENT: Middle sub-tasks build functionality
  VERIFY: Final sub-task ensures all tests pass
  UPDATE: Mark each sub-task complete as finished
</instructions>

</step>

<step number="7" subagent="test-runner" name="task_test_verification">

### Step 7: Task-Specific Test Verification

Use the test-runner subagent to run and verify only the tests specific to this parent task (not the full test suite) to ensure the feature is working correctly.

<focused_test_execution>
  <run_only>
    - All new tests written for this parent task
    - All tests updated during this task
    - Tests directly related to this feature
  </run_only>
  <skip>
    - Full test suite (done later in execute-tasks.md)
    - Unrelated test files
  </skip>
</focused_test_execution>

<final_verification>
  IF any test failures:
    - Debug and fix the specific issue
    - Re-run only the failed tests
  ELSE:
    - Confirm all task tests passing
    - Ready to proceed
</final_verification>

<instructions>
  ACTION: Use test-runner subagent
  REQUEST: "Run tests for [this parent task's test files]"
  WAIT: For test-runner analysis
  PROCESS: Returned failure information
  VERIFY: 100% pass rate for task-specific tests
  CONFIRM: This feature's tests are complete
</instructions>

</step>

<step number="8" name="ai_tool_problem_solving">

### Step 8: AI Tool Problem Solving (MANDATORY)

**MANDATORY REQUIREMENT: When encountering ANY problem, test failure, or issue, ALWAYS use sequential-thinking and context7 to plan the solution and ensure most updated information.**

<sequential_thinking_requirement>
  **MANDATORY**: Use sequential-thinking MCP to:
  - Break down the problem into logical steps
  - Identify root causes and dependencies
  - Plan the optimal solution approach
  - Consider potential side effects and risks
  - Outline step-by-step debugging strategy
</sequential_thinking_requirement>

<context7_requirement>
  **MANDATORY**: Use Context7 MCP to:
  - Fetch latest documentation for any libraries/frameworks involved
  - Get current best practices for the specific problem type
  - Research optimal solutions and patterns
  - Verify any API or tool usage is current
  - Find authoritative troubleshooting guides
</context7_requirement>

<problem_solving_output>
  - Sequential-thinking analysis of the problem
  - Context7 research results for current solutions
  - Comprehensive solution plan with step-by-step approach
  - Risk assessment and mitigation strategies
  - Success criteria and validation methods
</problem_solving_output>

<instructions>
  **ACTION**: ALWAYS use sequential-thinking MCP when ANY problem occurs
  **ACTION**: ALWAYS use Context7 MCP for current documentation and solutions
  **REQUIREMENT**: No problem-solving proceeds without both AI tool outputs
  **OUTPUT**: Comprehensive solution plan based on AI tool analysis
</instructions>

</step>

<step number="9" name="task_status_updates">

### Step 9: Task Status Updates

Update the tasks.md file immediately after completing each task to track progress.

<update_format>
  <completed>- [x] Task description</completed>
  <incomplete>- [ ] Task description</incomplete>
  <blocked>
    - [ ] Task description
    ⚠️ Blocking issue: [DESCRIPTION]
  </blocked>
</update_format>

<todo_list_standards>
  **Todo List Format Standards:**
  - **Incomplete tasks**: `- [ ] Task description`
  - **Completed tasks**: `- [x] Task description` 
  - **Never use**: `