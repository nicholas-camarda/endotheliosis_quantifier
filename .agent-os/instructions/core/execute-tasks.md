---
description: Rules to initiate execution of a set of tasks using Agent OS
globs:
alwaysApply: false
version: 1.0
encoding: UTF-8
---

# Task Execution Rules

## Overview

Initiate execution of one or more tasks for a given spec.

**TURN LIMITATION:**
- **DEFAULT**: Agent completes only 1 parent task (with 3-5 subtasks) per turn
- **EXCEPTION**: Continue only if user explicitly requests "continue" or "keep going"
- **PURPOSE**: Prevent overwhelming responses and maintain user control over execution pace

<pre_flight_check>
  EXECUTE: @~/.agent-os/instructions/meta/pre-flight.md
  NOTE: Path conventions — Home: @~/.agent-os, Project: @.agent-os
  
  **MANDATORY BEST-PRACTICES CHECK:**
  ALWAYS READ @~/.agent-os/standards/best-practices.md FIRST before any other action
  This file contains the core guidance for all agent behavior and routing
  
  **MANDATORY PROJECT CONTEXT CHECK:**
  ALWAYS READ @.agent-os/project-context.md FIRST to get key project information (current workspace)
  This file contains dataset names, function names, file paths, and other critical details
  
  WEB_SEARCH_CHECK:
    IF uncertain OR current info required:
      USE: @Web to fetch latest official docs/best practices (never curl)
  MCP_CONTEXT7_CHECK:
    WHEN: docs/api/library/examples/usage are needed OR library usage/version is uncertain
    USE: Context7 (resolve-library-id → get-library-docs) to fetch authoritative docs
  SEQUENTIAL_THINKING:
    ACTION: Use sequential-thinking MCP to plan the sequence across multiple tasks and define success criteria per parent task
  
  **ROUTING GUIDANCE:**
  After reading best-practices.md, use it to identify which specific helper files to consult:
  - For testing: @~/.agent-os/standards/testing-standards.md
  - For code style: @~/.agent-os/standards/code-style/[language]-style.md
  - For methodology: @~/.agent-os/standards/methodology.md
  - For tech stack: @~/.agent-os/standards/tech-stack.md
</pre_flight_check>

<process_flow>

<step number="1" name="spec_and_task_verification">

### Step 1: Spec and Task Verification

**ALWAYS check the spec and tasks before starting any new task or subtask.**

<verification_requirements>
  <best_practices_check>
    - **MANDATORY**: Read @~/.agent-os/standards/best-practices.md to refresh core guidance
    - Use best-practices.md to identify which specific helper files to consult
    - Ensure all relevant standards are loaded for current task context
  </best_practices_check>
  <spec_check>
    - Load and review the current spec to understand the full context
    - Verify the spec is up-to-date with any recent changes
    - Confirm the current development state and next steps
  </spec_check>
  <task_check>
    - Review the tasks.md file to understand the current task breakdown
    - Identify which tasks are completed, in progress, or pending
    - Confirm the specific task or subtask to be executed
  </task_check>
  <context_refresh>
    - Ensure all relevant spec files are loaded and current
    - Update understanding of the project state before proceeding
  </context_refresh>
</verification_requirements>

<instructions>
  ACTION: Always verify spec and tasks before proceeding
  REQUIREMENT: Load spec files and tasks.md to refresh context
  PURPOSE: Ensure accurate understanding of current state and requirements
  MANDATORY: This step cannot be skipped
</instructions>

</step>

<step number="2" name="task_assignment">

### Step 2: Task Assignment

Identify which tasks to execute from the spec (using spec_srd_reference file path and optional specific_tasks array), defaulting to the next uncompleted parent task if not specified.

<task_selection>
  <explicit>user specifies exact task(s)</explicit>
  <implicit>find next uncompleted task in tasks.md</implicit>
</task_selection>

<instructions>
  ACTION: Identify task(s) to execute
  DEFAULT: Select next uncompleted parent task if not specified
  CONFIRM: Task selection with user
</instructions>

</step>

<step number="3" subagent="context-fetcher" name="context_analysis">

### Step 3: Context Analysis

Use the context-fetcher subagent to gather minimal context for task understanding by always loading spec tasks.md, and conditionally loading @~/.agent-os/product/mission-lite.md, spec-lite.md, and sub-specs/technical-spec.md if not already in context.

<instructions>
  ACTION: Use context-fetcher subagent to:
    - REQUEST: "Get product pitch from mission-lite.md"
    - REQUEST: "Get spec summary from spec-lite.md"
    - REQUEST: "Get technical approach from technical-spec.md"
  PROCESS: Returned information
</instructions>


<context_gathering>
  <essential_docs>
    - tasks.md for task breakdown
  </essential_docs>
  <conditional_docs>
    - mission-lite.md for product alignment
    - spec-lite.md for feature summary
    - technical-spec.md for implementation details
  </conditional_docs>
</context_gathering>

</step>

<step number="4" name="development_server_check">

### Step 4: Check for Development Server

Check for any running development server and ask user permission to shut it down if found to prevent port conflicts.

<server_check_flow>
  <if_running>
    ASK user to shut down
    WAIT for response
  </if_running>
  <if_not_running>
    PROCEED immediately
  </if_not_running>
</server_check_flow>

<user_prompt>
  A development server is currently running.
  Should I shut it down before proceeding? (yes/no)
</user_prompt>

<instructions>
  ACTION: Check for running local development server
  CONDITIONAL: Ask permission only if server is running
  PROCEED: Immediately if no server detected
</instructions>

</step>

<step number="5" subagent="git-workflow" name="git_branch_management">

### Step 5: Git Branch Management

Use the git-workflow subagent to manage git branches to ensure proper isolation by creating or switching to the appropriate branch for the spec.

<instructions>
  ACTION: Use git-workflow subagent
  REQUEST: "Check and manage branch for spec: [SPEC_FOLDER]
            - Create branch if needed
            - Switch to correct branch
            - Handle any uncommitted changes"
  WAIT: For branch setup completion
</instructions>

<branch_naming>
  <source>spec folder name</source>
  <format>exclude date prefix</format>
  <example>
    - folder: 2025-03-15-password-reset
    - branch: password-reset
  </example>
</branch_naming>

</step>

<step number="6" name="task_execution_loop">

### Step 6: Task Execution Loop

Execute all assigned parent tasks and their subtasks using @~/.agent-os/instructions/core/execute-task.md instructions, continuing until all tasks are complete.

<execution_flow>
  LOAD @~/.agent-os/instructions/core/execute-task.md ONCE

  **TURN LIMITATION:**
  - **DEFAULT**: Complete only 1 parent task (with 3-5 subtasks) per turn
  - **EXCEPTION**: Continue only if user explicitly requests "continue" or "keep going"
  - **PURPOSE**: Prevent overwhelming responses and allow user control

  FOR each parent_task assigned in Step 1:
    **BEFORE each task:**
    - READ @~/.agent-os/standards/best-practices.md to refresh guidance
    - Use best-practices.md to identify relevant helper files for current task
    - Load appropriate standards files (testing, code-style, methodology, etc.)
    
    EXECUTE instructions from execute-task.md with:
      - parent_task_number
      - all associated subtasks
    WAIT for task completion
    UPDATE tasks.md status
    
    **AFTER completing 1 parent task:**
    - STOP execution unless user explicitly requests continuation
    - Provide summary of what was accomplished
    - Ask user if they want to continue with next task
    - End turn after 1 parent task completion
  END FOR
</execution_flow>

<loop_logic>
  <continue_conditions>
    - User explicitly requests "continue" or "keep going"
    - User specifically asks for more tasks to be completed
  </continue_conditions>
  <exit_conditions>
    - **DEFAULT**: After completing 1 parent task (with 3-5 subtasks)
    - All assigned tasks marked complete
    - User requests early termination
    - Blocking issue prevents continuation
    - User does not explicitly request continuation
  </exit_conditions>
</loop_logic>

<task_status_check>
  AFTER each task execution:
    CHECK tasks.md for remaining tasks
    UPDATE spec with any changes made during task execution
    
    **TURN LIMITATION ENFORCEMENT:**
    IF 1 parent task completed (with 3-5 subtasks):
      - STOP execution (default behavior)
      - Provide summary of accomplishments
      - Ask user if they want to continue
      - End turn unless user explicitly requests continuation
    ELSE IF all assigned tasks complete:
      PROCEED to next step
    ELSE IF user explicitly requests continuation:
      CONTINUE with next task
    ELSE:
      STOP and end turn (default behavior)
</task_status_check>

<instructions>
  ACTION: Load execute-task.md instructions once at start
  REUSE: Same instructions for each parent task iteration
  **TURN LIMITATION**: Complete only 1 parent task per turn unless user explicitly requests continuation
  LOOP: Through assigned parent tasks (limited to 1 per turn by default)
  **BEST-PRACTICES**: Check @~/.agent-os/standards/best-practices.md before each task
  **ROUTING**: Use best-practices.md to identify which helper files to load for current task
  UPDATE: Task status after each completion
  UPDATE: Spec after each turn to reflect any changes made
  VERIFY: All tasks complete before proceeding
  HANDLE: Blocking issues appropriately
  **END TURN**: After 1 parent task completion (default behavior)
</instructions>

</step>

<step number="7" name="spec_update_after_turn">

### Step 7: Spec Update After Each Turn

**CRITICAL: After each task execution turn, the agent MUST update the spec to reflect any changes made.**

<spec_update_requirements>
  <mandatory_update>
    - **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure compliance
    - Update spec files with any code changes, new functions, or modifications
    - Document any new dependencies, configurations, or setup requirements
    - Update technical approach if implementation details changed
    - Reflect any new insights or discoveries from the development process
  </mandatory_update>
  <update_timing>
    - AFTER each task execution turn
    - BEFORE proceeding to the next task
    - IMMEDIATELY after any significant changes
  </update_timing>
  <update_scope>
    - Code changes and new functionality
    - Dependencies and package requirements
    - Configuration and setup instructions
    - Testing approach and test data
    - Documentation and comments
  </update_scope>
</spec_update_requirements>

<instructions>
  ACTION: Update spec after each turn
  REQUIREMENT: Document all changes made during task execution
  PURPOSE: Keep spec current and accurate for future development
  TIMING: After each task turn, before proceeding
</instructions>

</step>

<step number="8" subagent="test-runner" name="test_suite_verification">

### Step 8: Run All Tests

Use the test-runner subagent to run the entire test suite to easily check if changes have fundamentally altered the pipeline. Fix any failures until all tests pass.

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure testing compliance
  ACTION: Use test-runner subagent
  REQUEST: "Run the full test suite"
  WAIT: For test-runner analysis
  PROCESS: Fix any reported failures (using AI tools for planning)
  REPEAT: Until all tests pass
  **MANDATORY**: Use sequential-thinking and context7 for ALL problem-solving
</instructions>

<test_execution>
  <order>
    1. Run entire test suite to check for pipeline changes
    2. Fix any failures that indicate broken functionality
  </order>
  <requirement>100% pass rate</requirement>
  <purpose>Quick validation that changes haven't fundamentally altered the pipeline</purpose>
</test_execution>

<failure_handling>
  <action>troubleshoot and fix</action>
  <priority>before proceeding</priority>
  <ai_tool_requirement>
    **MANDATORY**: Use sequential-thinking and context7 for ALL troubleshooting
    **NO EXCEPTIONS**: Every problem requires AI tool analysis before fixing
  </ai_tool_requirement>
</failure_handling>

</step>

<step number="9" name="ai_tool_problem_solving">

### Step 9: AI Tool Problem Solving (MANDATORY)

**MANDATORY REQUIREMENT: When encountering ANY problem, test failure, or issue during test execution, ALWAYS use sequential-thinking and context7 to plan the solution and ensure most updated information.**

<sequential_thinking_requirement>
  **MANDATORY**: Use sequential-thinking MCP to:
  - Break down the test failure or problem into logical steps
  - Identify root causes and dependencies
  - Plan the optimal debugging approach
  - Consider potential side effects and risks
  - Outline step-by-step troubleshooting strategy
</sequential_thinking_requirement>

<context7_requirement>
  **MANDATORY**: Use Context7 MCP to:
  - Fetch latest documentation for any libraries/frameworks involved
  - Get current best practices for the specific test failure type
  - Research optimal debugging solutions and patterns
  - Verify any API or tool usage is current
  - Find authoritative troubleshooting guides for test failures
</context7_requirement>

<problem_solving_output>
  - Sequential-thinking analysis of the test failure/problem
  - Context7 research results for current debugging solutions
  - Comprehensive debugging plan with step-by-step approach
  - Risk assessment and mitigation strategies
  - Success criteria and validation methods
</problem_solving_output>

<instructions>
  **ACTION**: ALWAYS use sequential-thinking MCP when ANY test failure or problem occurs
  **ACTION**: ALWAYS use Context7 MCP for current documentation and debugging solutions
  **REQUIREMENT**: No problem-solving proceeds without both AI tool outputs
  **OUTPUT**: Comprehensive debugging plan based on AI tool analysis
</instructions>

</step>

<step number="10" subagent="git-workflow" name="git_workflow">

### Step 10: Git Workflow (Prepare Only)

Prepare commit message, PR title/body, and command list. Do NOT execute git commands without explicit user approval.

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure git workflow compliance
  ACTION: Use git-workflow subagent
  REQUEST: "Prepare git workflow artifacts for [SPEC_NAME] feature (do not execute):
            - Draft commit message (no emojis/backticks; exclude .gitignore/.agent-os changes by default)
            - Draft PR title and description
            - List git commands to run (commit/push/PR)"
  OUTPUT: Present artifacts and await explicit user approval before running any commands
</instructions>

<commit_process>
  <commit>
    <message>draft descriptive summary (prepared, not executed)</message>
    <format>conventional commits if applicable</format>
  </commit>
  <push>
    <target>spec branch</target>
    <remote>origin</remote>
    <note>commands listed only; do not execute without approval</note>
  </push>
  <pull_request>
    <title>draft PR title</title>
    <description>draft functionality recap</description>
  </pull_request>
</commit_process>

</step>

<step number="11" name="roadmap_progress_check">

### Step 11: Roadmap Progress Check (Conditional)

Check @~/.agent-os/product/roadmap.md (if not in context) and update roadmap progress only if the executed tasks may have completed a roadmap item and the spec completes that item.

<conditional_execution>
  <preliminary_check>
    EVALUATE: Did executed tasks potentially complete a roadmap item?
    IF NO:
      SKIP this entire step
      PROCEED to step 9
    IF YES:
      CONTINUE with roadmap check
  </preliminary_check>
</conditional_execution>

<conditional_loading>
  IF roadmap.md NOT already in context:
    LOAD @~/.agent-os/product/roadmap.md
  ELSE:
    SKIP loading (use existing context)
</conditional_loading>

<roadmap_criteria>
  <update_when>
    - spec fully implements roadmap feature
    - all related tasks completed
    - tests passing
  </update_when>
  <caution>only mark complete if absolutely certain</caution>
</roadmap_criteria>

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure roadmap compliance
  ACTION: First evaluate if roadmap check is needed
  SKIP: If tasks clearly don't complete roadmap items
  CHECK: If roadmap.md already in context
  LOAD: Only if needed and not in context
  EVALUATE: If current spec completes roadmap goals
  UPDATE: Mark roadmap items complete if applicable
  VERIFY: Certainty before marking complete
</instructions>

</step>

<step number="12" name="completion_notification">

### Step 12: Task Completion Notification

Play a system sound to alert the user that tasks are complete.

<notification_command>
  afplay /System/Library/Sounds/Glass.aiff
</notification_command>

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure completion compliance
  ACTION: Play completion sound
  PURPOSE: Alert user that task is complete
</instructions>

</step>

<step number="12" name="spec_completion_check">

### Step 12: Spec Completion Check

**Important: Check if spec is complete and move to completed-specs if all tasks are done**

Check if all tasks in the spec are completed and move the spec to the completed-specs directory if appropriate.

<completion_check>
  <verification_process>
    1. Check tasks.md to see if all tasks are marked as complete
    2. Verify that all parent tasks and subtasks have [x] status
    3. Confirm no incomplete tasks remain
    4. Only proceed with moving if spec is truly complete
  </verification_process>
  <completion_criteria>
    - All parent tasks marked as complete [x]
    - All subtasks marked as complete [x]
    - No blocking issues (⚠️) remaining
    - All tests passing
    - Implementation fully functional
  </completion_criteria>
</completion_check>

<spec_movement_process>
  <if_complete>
    IF all_tasks_complete:
      1. Create .agent-os/specs/completed-specs/ directory if it doesn't exist
      2. Move spec folder to .agent-os/specs/completed-specs/
      3. Preserve all files and folder structure
      4. Update any references to the spec location
      5. Confirm successful movement
  </if_complete>
  <if_incomplete>
    IF any_tasks_incomplete:
      - Do not move spec
      - Continue with completion summary
      - Note that spec is not yet complete
  </if_incomplete>
</spec_movement_process>

<movement_commands>
  ```bash
  # Create completed-specs directory if it doesn't exist
  mkdir -p .agent-os/specs/completed-specs
  
  # Move completed spec to completed-specs directory
  mv .agent-os/specs/YYYY-MM-DD-spec-name .agent-os/specs/completed-specs/
  
  # Verify successful movement
  ls -la .agent-os/specs/completed-specs/YYYY-MM-DD-spec-name/
  ```
</movement_commands>

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure completion check compliance
  ACTION: Check if all tasks are complete
  VERIFY: All tasks marked as [x] in tasks.md
  MOVE: Spec to completed-specs if truly complete
  PRESERVE: All files and structure during movement
  CONFIRM: Successful movement
</instructions>

</step>

<step number="13" name="completion_summary">

### Step 13: Completion Summary

Create a structured summary message with emojis showing what was done, any issues, testing instructions, PR link, and spec completion status.

<summary_template>
  ## ✅ What's been done

  1. **[FEATURE_1]** - [ONE_SENTENCE_DESCRIPTION]
  2. **[FEATURE_2]** - [ONE_SENTENCE_DESCRIPTION]

  ## ⚠️ Issues encountered

  [ONLY_IF_APPLICABLE]
  - **[ISSUE_1]** - [DESCRIPTION_AND_REASON]

  ## 👀 Ready to test in browser

  [ONLY_IF_APPLICABLE]
  1. [STEP_1_TO_TEST]
  2. [STEP_2_TO_TEST]

  ## 📦 Pull Request

  View PR: [GITHUB_PR_URL]

  ## 📁 Spec Status

  [SPEC_COMPLETION_STATUS]
  - **Complete**: Spec moved to .agent-os/specs/completed-specs/
  - **In Progress**: Spec remains in .agent-os/specs/ for continued work
</summary_template>

<summary_sections>
  <required>
    - functionality recap
    - pull request info
    - spec completion status
  </required>
  <conditional>
    - issues encountered (if any)
    - testing instructions (if testable in browser)
  </conditional>
</summary_sections>

<instructions>
  **FIRST**: Read @~/.agent-os/standards/best-practices.md to ensure summary compliance
  ACTION: Create comprehensive summary
  INCLUDE: All required sections
  ADD: Conditional sections if applicable
  FORMAT: Use emoji headers for scannability
  NOTE: Spec completion status and location
</instructions>

</step>

</process_flow>

## Error Handling

<error_protocols>
  <blocking_issues>
    - document in tasks.md
    - mark with ⚠️ emoji
    - include in summary
  </blocking_issues>
  <test_failures>
    - fix before proceeding
    - never commit broken tests
  </test_failures>
  <technical_roadblocks>
    - attempt 3 approaches
    - document if unresolved
    - seek user input
  </technical_roadblocks>
</error_protocols>

<final_checklist>
  <verify>
    - [ ] **BEST-PRACTICES**: @~/.agent-os/standards/best-practices.md consulted
    - [ ] Task implementation complete
    - [ ] All tests passing
    - [ ] tasks.md updated
    - [ ] Code committed and pushed
    - [ ] Pull request created
    - [ ] Roadmap checked/updated
    - [ ] Spec completion status checked
    - [ ] Completed specs moved to .agent-os/specs/completed-specs/ if appropriate
    - [ ] Summary provided to user
  </verify>
</final_checklist>

<todo_list_standards>
  **Todo List Format Standards:**
  - **Incomplete tasks**: `- [ ] Task description`
  - **Completed tasks**: `- [x] Task description` 
  - **Never use**: `- [ ] Task description ✅ **COMPLETED**`
  - **Update immediately**: Change `[ ]` to `[x]` when task is done
  - **Keep clean**: No completion text inside checkboxes
</todo_list_standards>

## Execution Standards

<standards>
  <follow>
    - **PRIMARY**: @~/.agent-os/standards/best-practices.md (consult FIRST for routing guidance)
    - @~/.agent-os/standards/testing-standards.md
    - @~/.agent-os/standards/code-style/r-style.md
    - @~/.agent-os/standards/code-style/javascript-style.md
    - @~/.agent-os/standards/code-style/html-style.md
    - @~/.agent-os/standards/code-style/css-style.md
    - @~/.agent-os/product/tech-stack.md
  </follow>
  <maintain>
    - Consistency with product mission
    - Alignment with roadmap
    - Technical coherence
    - Follow backwards compatibility anti-pattern from best-practices.md
  </maintain>
  <create>
    - Comprehensive documentation
    - Clear implementation path
    - Testable outcomes
    - Testing procedures in every spec
    - Clean implementation without legacy code excuses
  </create>
</standards>
