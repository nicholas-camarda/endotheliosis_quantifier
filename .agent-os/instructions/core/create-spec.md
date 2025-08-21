---
description: Spec Creation Rules for Agent OS
globs:
alwaysApply: false
version: 1.1
encoding: UTF-8
---

# Spec Creation Rules

## Overview

Generate detailed feature specifications aligned with product roadmap and mission.

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
    **MANDATORY**: ALWAYS use Context7 for current documentation and best practices
    WHEN: docs/api/library/examples/usage are needed OR library usage/version is uncertain
    USE: Context7 (resolve-library-id → get-library-docs) to fetch authoritative docs
  SEQUENTIAL_THINKING:
    **MANDATORY**: ALWAYS use sequential-thinking MCP for problem analysis and planning
    ACTION: Use sequential-thinking MCP to outline plan, assumptions, risks, and success criteria before major steps
  STANDARDS_INDEX_CHECK: If unsure which standard to consult, READ @~/.agent-os/standards/README.md to pick targets
  AI_TOOL_REQUIREMENT:
    **CRITICAL**: Every spec creation MUST use both sequential-thinking and context7
    **NO EXCEPTIONS**: AI tools required for planning and current information gathering
</pre_flight_check>

## Testing Standards Reference

### Testing Approach
Follow the appropriate testing standards for your project:

#### **General Testing Standards**
**Reference:** `standards/testing-standards.md` - Contains testing principles including:
- Test organization and execution
- Framework-specific commands
- Core testing principles

#### **Language-Specific Testing Standards**
**Reference:** `standards/code-style/` directory for language-specific testing standards:
- `r-style.md` - R language testing standards
- `python-style.md` - Python testing standards
- `javascript-style.md` - JavaScript testing standards
- `html-style.md` - HTML testing standards
- `css-style.md` - CSS testing standards

### Testing Implementation
- Use appropriate testing frameworks for your language
- Follow language-specific testing protocols
- Test actual behavior with realistic data
- Keep tests simple and focused

---

<process_flow>

<step number="1" subagent="context-fetcher" name="spec_initiation">

### Step 1: Spec Initiation

Use the context-fetcher subagent to identify spec initiation method by either finding the next uncompleted roadmap item when user asks "what's next?" or accepting a specific spec idea from the user.

<option_a_flow>
  <trigger_phrases>
    - "what's next?"
  </trigger_phrases>
  <actions>
    1. CHECK @~/.agent-os/product/roadmap.md
    2. FIND next uncompleted item
    3. SUGGEST item to user
    4. WAIT for approval
  </actions>
</option_a_flow>

<option_b_flow>
  <trigger>user describes specific spec idea</trigger>
  <accept>any format, length, or detail level</accept>
  <proceed>to context gathering</proceed>
</option_b_flow>

</step>

<step number="2" subagent="context-fetcher" name="context_gathering">

### Step 2: Context Gathering (Conditional)

Use the context-fetcher subagent to read @~/.agent-os/product/mission-lite.md and @~/.agent-os/product/tech-stack.md only if not already in context to ensure minimal context for spec alignment.

<conditional_logic>
  IF both mission-lite.md AND tech-stack.md already read in current context:
    SKIP this entire step
    PROCEED to step 3
  ELSE:
    READ only files not already in context:
      - mission-lite.md (if not in context)
      - tech-stack.md (if not in context)
    CONTINUE with context analysis
</conditional_logic>

<context_analysis>
  <mission_lite>core product purpose and value</mission_lite>
  <tech_stack>technical requirements</tech_stack>
</context_analysis>

</step>

<step number="3" subagent="context-fetcher" name="requirements_clarification">

### Step 3: Requirements Clarification

Use the context-fetcher subagent to clarify scope boundaries and technical considerations by asking numbered questions as needed to ensure clear requirements before proceeding.

<clarification_areas>
  <scope>
    - in_scope: what is included
    - out_of_scope: what is excluded (optional)
  </scope>
  <technical>
    - functionality specifics
    - UI/UX requirements
    - integration points
  </technical>
</clarification_areas>

<decision_tree>
  IF clarification_needed:
    ASK numbered_questions
    WAIT for_user_response
  ELSE:
    PROCEED to_date_determination
</decision_tree>

</step>

<step number="4" name="ai_tool_planning">

### Step 4: AI Tool Planning (MANDATORY)

**MANDATORY REQUIREMENT: After clarifying requirements, ALWAYS use sequential-thinking and context7 to plan the spec creation and ensure most updated information.**

<sequential_thinking_requirement>
  **MANDATORY**: Use sequential-thinking MCP to:
  - Break down the spec creation into logical steps
  - Identify potential challenges and dependencies
  - Plan the optimal approach for spec development
  - Consider impact on existing systems and architecture
  - Outline success criteria and validation steps
</sequential_thinking_requirement>

<context7_requirement>
  **MANDATORY**: Use Context7 MCP to:
  - Fetch latest documentation for any libraries/frameworks mentioned
  - Get current best practices for spec creation and testing approaches
  - Ensure technical requirements are based on current standards
  - Verify any API or tool usage is up-to-date
  - Research optimal patterns for the specific feature type
</context7_requirement>

<planning_output>
  - Sequential-thinking analysis of the spec creation process
  - Context7 research results for current best practices
  - Comprehensive spec creation plan with step-by-step approach
  - Risk assessment and mitigation strategies
  - Success criteria and validation methods
</planning_output>

<instructions>
  **ACTION**: ALWAYS use sequential-thinking MCP for spec creation planning
  **ACTION**: ALWAYS use Context7 MCP for current documentation and best practices
  **REQUIREMENT**: No spec creation proceeds without both AI tool outputs
  **OUTPUT**: Comprehensive plan based on AI tool analysis
</instructions>

</step>

<step number="5" name="function_analysis">

### Step 5: Function Analysis & Context

Perform function analysis to understand current codebase state and identify relevant existing functions before creating new specifications.

<analysis_requirements>
  <dependency_analysis>
    - Review current architecture and module organization
    - Identify potential integration points for new functionality
    - Check for existing patterns that should be followed
    - Assess impact of new features on existing code
  </dependency_analysis>
  <existing_functionality_check>
    - Check for existing functions that could handle the new spec requirements
    - Identify functions that could be extended rather than creating new ones
    - Assess whether existing functions can be modified to meet spec needs
    - Document any existing functionality that could be leveraged
  </existing_functionality_check>
  <complexity_threshold_assessment>
    - Evaluate complexity of new spec requirements against existing functions
    - Determine if new functionality exceeds complexity threshold for function modification
    - If complexity is low: recommend extending existing functions
    - If complexity is high: recommend creating new functions
    - Document rationale for function creation vs. modification approach
  </complexity_threshold_assessment>
  <context_gathering>
    - Identify existing functions that new spec might replace or extend
    - Document current patterns and conventions to maintain consistency
    - Check for existing utilities or helpers that could be reused
    - Assess technical debt or refactoring opportunities
  </context_gathering>
</analysis_requirements>

<execution_instructions>
  ANALYZE: Current codebase state relevant to new specification
  CHECK: Existing functionality that could handle new spec requirements
  EVALUATE: Complexity threshold for function creation vs. modification
  DOCUMENT: Existing functions and patterns that inform new spec
  IDENTIFY: Integration points and potential conflicts
</execution_instructions>

<output_requirements>
  - Summary of relevant existing functions
  - Existing functionality assessment and recommendations
  - Complexity threshold evaluation and function approach rationale
  - Integration recommendations for new specification
</output_requirements>

</step>

<step number="6" name="date_determination">

### Step 6: Date Determination

Determine accurate date for folder naming by creating a temporary file to extract timestamp in YYYY-MM-DD format, falling back to asking user if needed.

<date_determination_process>
  <primary_method>
    <name>File System Timestamp</name>
    <process>
      1. CREATE directory if not exists: .agent-os/specs/
      2. CREATE temporary file: .agent-os/specs/.date-check
      3. READ file creation timestamp from filesystem
      4. EXTRACT date in YYYY-MM-DD format
      5. DELETE temporary file
      6. STORE date in variable for folder naming
    </process>
  </primary_method>

  <fallback_method>
    <trigger>if file system method fails</trigger>
    <name>User Confirmation</name>
    <process>
      1. STATE: "I need to confirm today's date for the spec folder"
      2. ASK: "What is today's date? (YYYY-MM-DD format)"
      3. WAIT for user response
      4. VALIDATE format matches YYYY-MM-DD
      5. STORE date for folder naming
    </process>
  </fallback_method>
</date_determination_process>

<validation>
  <format_check>^\d{4}-\d{2}-\d{2}$</format_check>
  <reasonableness_check>
    - year: 2024-2030
    - month: 01-12
    - day: 01-31
  </reasonableness_check>
</validation>

<error_handling>
  IF date_invalid:
    USE fallback_method
  IF both_methods_fail:
    ERROR "Unable to determine current date"
</error_handling>

</step>

<step number="7" subagent="file-creator" name="spec_folder_creation">

### Step 7: Spec Folder Creation

Use the file-creator subagent to create directory: .agent-os/specs/YYYY-MM-DD-spec-name/ using the date from step 4.

**Important: Preserve existing spec files**

<folder_naming>
  <format>YYYY-MM-DD-spec-name</format>
  <date>use stored date from step 4</date>
  <name_constraints>
    - max_words: 5
    - style: kebab-case
    - descriptive: true
  </name_constraints>
</folder_naming>

<example_names>
  - 2025-03-15-password-reset-flow
  - 2025-03-16-user-profile-dashboard
  - 2025-03-17-api-rate-limiting
</example_names>

<file_preservation_check>
  <before_creation>
    - Check if spec folder already exists
    - Check if spec.md already exists in folder
    - If files exist, read existing content first
    - Preserve existing work
  </before_creation>
  <if_files_exist>
    - Read existing spec content
    - Work with existing content
    - Ask user for permission before any destructive operations
    - Preserve all existing work
  </if_files_exist>
</file_preservation_check>

</step>

<step number="8" subagent="file-creator" name="create_spec_md">

### Step 8: Create spec.md

**Important: Preserve existing spec files**

Use the file-creator subagent to create the file: .agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name.md using this template:

**Note:** The spec file is named the same as the folder for ease of calling with "@" commands.

<file_preservation_check>
  <before_creation>
    - Check if YYYY-MM-DD-spec-name.md already exists
    - If exists, read existing content first
    - Preserve existing work
    - Work with existing content if present
  </before_creation>
  <if_file_exists>
    - Read existing YYYY-MM-DD-spec-name.md content
    - Preserve existing work
    - Ask user for permission before any destructive operations
    - Work with existing content rather than replacing it
  </if_file_exists>
</file_preservation_check>

<file_template>
  <header>
    # Spec Requirements Document

    > Spec: [SPEC_NAME]
    > Created: [CURRENT_DATE]
    > Status: Planning
  </header>
  <required_sections>
    - Overview
    - User Stories
    - Spec Scope
    - Out of Scope
    - Expected Deliverable
    - Testing Strategy
  </required_sections>
</file_template>

<section name="overview">
  <template>
    ## Overview

    [1-2_SENTENCE_GOAL_AND_OBJECTIVE]
  </template>
  <constraints>
    - length: 1-2 sentences
    - content: goal and objective
  </constraints>
  <example>
    Implement a secure password reset functionality that allows users to regain account access through email verification. This feature will reduce support ticket volume and improve user experience by providing self-service account recovery.
  </example>
</section>

<section name="user_stories">
  <template>
    ## User Stories

    ### [STORY_TITLE]

    As a [USER_TYPE], I want to [ACTION], so that [BENEFIT].

    [DETAILED_WORKFLOW_DESCRIPTION]
  </template>
  <constraints>
    - count: 1-3 stories
    - include: workflow and problem solved
    - format: title + story + details
  </constraints>
</section>

<section name="spec_scope">
  <template>
    ## Spec Scope

    1. **[FEATURE_NAME]** - [ONE_SENTENCE_DESCRIPTION]
    2. **[FEATURE_NAME]** - [ONE_SENTENCE_DESCRIPTION]
  </template>
  <constraints>
    - count: 1-5 features
    - format: numbered list
    - description: one sentence each
  </constraints>
</section>

<section name="out_of_scope">
  <template>
    ## Out of Scope

    - [EXCLUDED_FUNCTIONALITY_1]
    - [EXCLUDED_FUNCTIONALITY_2]
  </template>
  <purpose>explicitly exclude functionalities</purpose>
</section>

<section name="expected_deliverable">
  <template>
    ## Expected Deliverable

    1. [TESTABLE_OUTCOME_1]
    2. [TESTABLE_OUTCOME_2]

    ## Implementation Reference

    **All implementation details are documented in tasks.md**
    - See tasks.md for detailed implementation steps
    - See tasks.md for technical specifications
    - See tasks.md for testing procedures
  </template>
  <constraints>
    - count: 1-3 expectations
    - focus: browser-testable outcomes
    - Include reference to tasks.md for implementation details
    - Never include detailed implementation steps in spec
  </constraints>
</section>

<section name="testing_strategy">
  <template>
    ## Testing Strategy

    ### Testing Approach
    This spec follows the testing standards from:
    - `standards/testing-standards.md` - General testing principles
    - `standards/code-style.md` - Language-agnostic testing standards
    - `standards/code-style/[language]-style.md` - Language-specific testing standards

    ### Testing Implementation Reference
    **All detailed testing procedures are documented in tasks.md**
    - See tasks.md for detailed testing steps and procedures
    - See tasks.md for specific test file creation instructions
    - See tasks.md for testing framework commands and validation steps
    - See tasks.md for error handling and performance testing procedures

    ### Testing Standards Reference
    **This spec follows the testing protocols from:**
    - `standards/testing-standards.md` - Universal testing principles
    - `standards/code-style.md` - Language-agnostic testing standards
    - `standards/code-style/[language]-style.md` - Language-specific testing standards
  </template>
  <constraints>
    - This section should be included in every spec
    - Reference appropriate testing standards
    - Never include detailed testing steps in spec - reference tasks.md
  </constraints>
</section>

</step>

<step number="9" subagent="file-creator" name="create_spec_lite_md">

### Step 9: Create spec-lite.md

Use the file-creator subagent to create the file: .agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name-lite.md for the purpose of establishing a condensed spec for efficient AI context usage.

**Note:** The lite file is named with "-lite" suffix for easy reference.

<file_template>
  <header>
    # Spec Summary (Lite)
  </header>
</file_template>

<content_structure>
  <spec_summary>
    - source: Step 6 spec.md overview section
    - length: 1-3 sentences
    - content: core goal and objective of the feature
  </spec_summary>
</content_structure>

<content_template>
  [1-3_SENTENCES_SUMMARIZING_SPEC_GOAL_AND_OBJECTIVE]
</content_template>

<example>
  Implement secure password reset via email verification to reduce support tickets and enable self-service account recovery. Users can request a reset link, receive a time-limited token via email, and set a new password following security best practices.
</example>

</step>

<step number="10" subagent="file-creator" name="create_technical_spec">

### Step 10: Create Technical Specification

Use the file-creator subagent to create the file: sub-specs/technical-spec.md using this template:

<file_template>
  <header>
    # Technical Specification

    This is the technical specification for the spec detailed in @~/.agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name.md
  </header>
</file_template>

<spec_sections>
  <technical_requirements>
    - functionality details
    - UI/UX specifications
    - integration requirements
    - performance criteria
  </technical_requirements>
  <external_dependencies_conditional>
    - only include if new dependencies needed
    - new libraries/packages
    - justification for each
    - version requirements
  </external_dependencies_conditional>
</spec_sections>

<example_template>
  ## Technical Requirements

  - [SPECIFIC_TECHNICAL_REQUIREMENT]
  - [SPECIFIC_TECHNICAL_REQUIREMENT]

  ## External Dependencies (Conditional)

  [ONLY_IF_NEW_DEPENDENCIES_NEEDED]
  - **[LIBRARY_NAME]** - [PURPOSE]
  - **Justification:** [REASON_FOR_INCLUSION]
</example_template>

<conditional_logic>
  IF spec_requires_new_external_dependencies:
    INCLUDE "External Dependencies" section
  ELSE:
    OMIT section entirely
</conditional_logic>

</step>

<step number="11" subagent="file-creator" name="create_database_schema">

### Step 11: Create Database Schema (Conditional)

Use the file-creator subagent to create the file: sub-specs/database-schema.md ONLY IF database changes needed for this task.

<decision_tree>
  IF spec_requires_database_changes:
    CREATE sub-specs/database-schema.md
  ELSE:
    SKIP this_step
</decision_tree>

<file_template>
  <header>
    # Database Schema

    This is the database schema implementation for the spec detailed in @~/.agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name.md
  </header>
</file_template>

<schema_sections>
  <changes>
    - new tables
    - new columns
    - modifications
    - migrations
  </changes>
  <specifications>
    - exact SQL or migration syntax
    - indexes and constraints
    - foreign key relationships
  </specifications>
  <rationale>
    - reason for each change
    - performance considerations
    - data integrity rules
  </rationale>
</schema_sections>

</step>

<step number="12" subagent="file-creator" name="create_api_spec">

### Step 12: Create API Specification (Conditional)

Use the file-creator subagent to create file: sub-specs/api-spec.md ONLY IF API changes needed.

<decision_tree>
  IF spec_requires_api_changes:
    CREATE sub-specs/api-spec.md
  ELSE:
    SKIP this_step
</decision_tree>

<file_template>
  <header>
    # API Specification

    This is the API specification for the spec detailed in @~/.agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name.md
  </header>
</file_template>

<api_sections>
  <routes>
    - HTTP method
    - endpoint path
    - parameters
    - response format
  </routes>
  <controllers>
    - action names
    - business logic
    - error handling
  </controllers>
  <purpose>
    - endpoint rationale
    - integration with features
  </purpose>
</api_sections>

<endpoint_template>
  ## Endpoints

  ### [HTTP_METHOD] [ENDPOINT_PATH]

  **Purpose:** [DESCRIPTION]
  **Parameters:** [LIST]
  **Response:** [FORMAT]
  **Errors:** [POSSIBLE_ERRORS]
</endpoint_template>

</step>

<step number="13" name="user_review">

### Step 13: User Review

Request user review of spec.md and all sub-specs files, waiting for approval or revision requests before proceeding to task creation.

<review_request>
  I've created the spec documentation:

  - Spec Requirements: @~/.agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name.md
- Spec Summary: @~/.agent-os/specs/YYYY-MM-DD-spec-name/YYYY-MM-DD-spec-name-lite.md
- Technical Spec: @~/.agent-os/specs/YYYY-MM-DD-spec-name/sub-specs/technical-spec.md
  [LIST_OTHER_CREATED_SPECS]

  **Note**: The spec includes testing procedures that follow the protocols from best-practices.md and appropriate code-style standards.

  Please review and let me know if any changes are needed before I create the task breakdown.
</review_request>

</step>

<step number="14" name="task_creation_reference">

### Step 14: Task Creation Reference

**Note: Task creation is now handled by the dedicated create-tasks command**

After spec approval, use the create-tasks command to generate the implementation task list:

<task_creation_instruction>
  **ACTION**: Use the create-tasks command to generate tasks.md
  **COMMAND**: create-tasks [spec-name]
  **PURPOSE**: Generate strategic, purposeful implementation tasks
  **OUTPUT**: tasks.md with focused, meaningful tasks that directly solve the core problem
</task_creation_instruction>

<task_creation_benefits>
  - Dedicated command for task generation
  - Strategic planning with sequential-thinking
  - Focus on purposeful, non-redundant tasks
  - Consistent task structure across all specs
  - Better separation of concerns
</task_creation_benefits>

</step>

<step number="15" name="file_verification">

### Step 15: File Verification

**Important: Verify all required files are created**

After creating all spec files, verify that all required files exist and are properly created.

<verification_requirements>
  <required_files>
    - `YYYY-MM-DD-spec-name.md` (main spec file)
    - `YYYY-MM-DD-spec-name-lite.md` (lite spec file)
    - `sub-specs/technical-spec.md` (if applicable)
    - `sub-specs/database-schema.md` (if applicable)
    - `sub-specs/api-spec.md` (if applicable)
  </required_files>
  <note>
    **Note**: tasks.md will be created separately using the create-tasks command
  </note>
  <verification_process>
    1. Verify main spec file exists and has content
    2. Verify lite spec file exists and has content
    3. Check conditional files if they should exist
    4. If any required files are missing, create them immediately
  </verification_process>
</verification_requirements>

<error_handling>
  IF main_spec_missing:
    CREATE main spec file immediately
    STATE "Main spec file was missing - created it now"
  IF lite_spec_missing:
    CREATE lite spec file immediately
    STATE "Lite spec file was missing - created it now"
  IF any_required_file_missing:
    CREATE missing files immediately
    STATE "Missing required files - created them now"
</error_handling>

<verification_confirmation>
  AFTER verification:
    STATE "All required spec files verified and created:"
    LIST: All created files
    CONFIRM: No missing files
    PROCEED: Only after all files are confirmed to exist
</verification_confirmation>

</step>

<step number="16" name="decision_documentation">

### Step 16: Decision Documentation (Conditional)

Evaluate strategic impact without loading decisions.md and update it only if there's significant deviation from mission/roadmap and user approves.

<conditional_reads>
  IF mission-lite.md NOT in context:
    USE: context-fetcher subagent
    REQUEST: "Get product pitch from mission-lite.md"
  IF roadmap.md NOT in context:
    USE: context-fetcher subagent
    REQUEST: "Get current development phase from roadmap.md"

  <manual_reads>
    <mission_lite>
      - IF NOT already in context: READ @~/.agent-os/product/mission-lite.md
      - IF already in context: SKIP reading
    </mission_lite>
    <roadmap>
      - IF NOT already in context: READ @~/.agent-os/product/roadmap.md
      - IF already in context: SKIP reading
    </roadmap>
    <decisions>
      - NEVER load decisions.md into context
    </decisions>
  </manual_reads>
</conditional_reads>

<decision_analysis>
  <review_against>
    - @~/.agent-os/product/mission-lite.md (conditional)
- @~/.agent-os/product/roadmap.md (conditional)
  </review_against>
  <criteria>
    - significantly deviates from mission in mission-lite.md
    - significantly changes or conflicts with roadmap.md
  </criteria>
</decision_analysis>

<decision_tree>
  IF spec_does_NOT_significantly_deviate:
    SKIP this entire step
    STATE "Spec aligns with mission and roadmap"
    PROCEED to step 16
  ELSE IF spec_significantly_deviates:
    EXPLAIN the significant deviation
    ASK user: "This spec significantly deviates from our mission/roadmap. Should I draft a decision entry?"
    IF user_approves:
      DRAFT decision entry
      UPDATE decisions.md
    ELSE:
      SKIP updating decisions.md
      PROCEED to step 16
</decision_tree>

<decision_template>
  ## [CURRENT_DATE]: [DECISION_TITLE]

  **ID:** DEC-[NEXT_NUMBER]
  **Status:** Accepted
  **Category:** [technical/product/business/process]
  **Related Spec:** @~/.agent-os/specs/YYYY-MM-DD-spec-name/

  ### Decision

  [DECISION_SUMMARY]

  ### Context

  [WHY_THIS_DECISION_WAS_NEEDED]

  ### Deviation

  [SPECIFIC_DEVIATION_FROM_MISSION_OR_ROADMAP]
</decision_template>

</step>

<step number="17" name="execution_readiness">

### Step 17: Execution Readiness Check

Evaluate readiness to begin implementation after completing all previous steps, presenting the first task summary and requesting user confirmation to proceed.

<readiness_summary>
  <present_to_user>
    - Spec name and description
    - First task summary from tasks.md
    - Estimated complexity/scope
    - Key deliverables for task 1
    - Testing procedures included in spec
    - All required files verified and created
  </present_to_user>
</readiness_summary>

<execution_prompt>
  PROMPT: "The spec planning is complete. All required files have been verified and created. The first task is:

  **Task 1:** [FIRST_TASK_TITLE]
  [BRIEF_DESCRIPTION_OF_TASK_1_AND_SUBTASKS]

  **Note**: This spec includes testing procedures that follow the protocols from best-practices.md and appropriate code-style standards. All testing will use appropriate testing frameworks for the language/framework.

  **Note**: All required spec files have been verified and created. Use the create-tasks command to generate the implementation task list.

  Would you like me to proceed with implementing Task 1? I will focus only on this first task and its subtasks unless you specify otherwise.

  Type 'yes' to proceed with Task 1, or let me know if you'd like to review or modify the plan first."
</execution_prompt>

<execution_flow>
  IF user_confirms_yes:
    REFERENCE: @~/.agent-os/instructions/core/execute-task.md
    FOCUS: Only Task 1 and its subtasks
    CONSTRAINT: Do not proceed to additional tasks without explicit user request
  ELSE:
    WAIT: For user clarification or modifications
</execution_flow>

</step>

</process_flow>

## Execution Standards

<standards>
  <follow>
    - @~/.agent-os/standards/testing-standards.md
    - @~/.agent-os/standards/code-style/r-style.md
    - @~/.agent-os/standards/code-style/python-style.md
    - @~/.agent-os/standards/code-style/javascript-style.md
    - @~/.agent-os/standards/code-style/html-style.md
    - @~/.agent-os/standards/code-style/css-style.md
    - @~/.agent-os/product/tech-stack.md
    - @~/.agent-os/standards/best-practices.md
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

<final_checklist>
  <verify>
    - [ ] Accurate date determined via file system
    - [ ] Spec folder created with correct date prefix
    - [ ] Existing spec files preserved - no overwriting
    - [ ] Existing work maintained if files already existed
    - [ ] YYYY-MM-DD-spec-name.md contains all required sections
    - [ ] Testing strategy included in YYYY-MM-DD-spec-name.md
    - [ ] Testing procedures reference appropriate code-style standards
    - [ ] Spec points to tasks.md for implementation details
    - [ ] No implementation details duplicated in spec
    - [ ] All applicable sub-specs created
    - [ ] User approved documentation
    - [ ] Cross-references added to YYYY-MM-DD-spec-name.md
    - [ ] Strategic decisions evaluated
    - [ ] All required files verified and created
    - [ ] YYYY-MM-DD-spec-name-lite.md exists and has content
    - [ ] All conditional files created if applicable
    - [ ] Note: tasks.md will be created separately using create-tasks command
  </verify>
</final_checklist>

<todo_list_guidelines>
  **Todo List Best Practices:**
  - Use `- [ ]` for incomplete tasks
  - Use `- [x]` for completed tasks (NOT `- [ ] ✅ **COMPLETED**`)
  - Keep task descriptions concise and clear
  - Update status immediately when tasks are completed
  - Avoid adding completion text inside checkboxes
  - Follow TDD approach in task structure
  - Include testing steps in every task
</todo_list_guidelines>
