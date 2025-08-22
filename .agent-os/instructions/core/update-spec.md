---
description: Spec Update Rules for Agent OS
globs:
alwaysApply: false
version: 1.0
encoding: UTF-8
---

# Spec Update Rules

## Overview

Update existing specifications to reflect changes, new requirements, or corrections while maintaining consistency with product roadmap and mission.

**MANDATORY REQUIREMENT: Every spec update MUST include updating tasks.md to ensure implementation tasks stay synchronized with spec requirements.**

<pre_flight_check>
  EXECUTE: @~/.agent-os/instructions/meta/pre-flight.md
  NOTE: Path conventions — Home: @~/.agent-os, Project: @.agent-os
  BEST_PRACTICES_CHECK:
    **MANDATORY**: ALWAYS read @~/.agent-os/standards/best-practices.md first
    VERIFY: Following established best practices and coding standards
    REFERENCE: This file for all development decisions and approaches
  WEB_SEARCH_CHECK:
    IF uncertain OR current info required:
      USE: @Web to fetch latest official docs/best practices (never curl)
  MCP_CONTEXT7_CHECK:
    **MANDATORY**: ALWAYS use Context7 for current documentation and best practices
    WHEN: docs/api/library/examples/usage are needed OR library usage/version is uncertain
    USE: Context7 (resolve-library-id → get-library-docs) to fetch authoritative docs
  SEQUENTIAL_THINKING:
    **MANDATORY**: ALWAYS use sequential-thinking MCP for problem analysis and planning
    ACTION: Use sequential-thinking MCP to outline plan, assumptions, risks, and success criteria before major updates
  STANDARDS_INDEX_CHECK: If unsure which standard to consult, READ @~/.agent-os/standards/best-practices.md and @~/.agent-os/standards/testing-standards.md to pick targets
  AI_TOOL_REQUIREMENT:
    **CRITICAL**: Every spec update MUST use both sequential-thinking and context7
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

<step number="1" subagent="context-fetcher" name="spec_identification">

### Step 1: Spec Identification

Use the context-fetcher subagent to identify which spec to update based on user input and current context.

<identification_methods>
  <option_a_flow>
    <trigger_phrases>
      - "@update-spec"
      - "@"
      - "@current"
      - "@spec"
    </trigger_phrases>
    <actions>
      1. ANALYZE current conversation context
      2. IDENTIFY spec currently being discussed
      3. LOCATE spec file in .agent-os/specs/
      4. CONFIRM spec exists and is accessible
    </actions>
  </option_a_flow>

  <option_b_flow>
    <trigger_phrases>
      - "@spec-name"
      - "@path/to/spec"
      - "@YYYY-MM-DD-spec-name"
    </trigger_phrases>
    <actions>
      1. EXTRACT spec identifier from user input
      2. SEARCH .agent-os/specs/ for matching spec
      3. VALIDATE spec exists and is accessible
      4. CONFIRM with user if multiple matches found
    </actions>
  </option_b_flow>

  <option_c_flow>
    <trigger_phrases>
      - "@list"
      - "@specs"
      - "@available"
    </trigger_phrases>
    <actions>
      1. LIST all available specs in .agent-os/specs/
      2. DISPLAY spec names and creation dates
      3. ASK user to specify which spec to update
      4. PROCEED with selected spec
    </actions>
  </option_c_flow>
</identification_methods>

<instructions>
  ACTION: Use context-fetcher subagent to identify target spec
  VALIDATE: Spec exists and is accessible
  CONFIRM: Spec selection with user if needed
</instructions>

</step>

<step number="2" subagent="context-fetcher" name="spec_analysis">

### Step 2: Comprehensive Spec Analysis

**Important: Comprehensive spec analysis - check all sections systematically**

Use the context-fetcher subagent to read and analyze the complete spec file to understand current state and identify what needs updating.

<comprehensive_analysis>
  <mandatory_sections_check>
    - Read and analyze ALL sections of the spec
    - Check Problem Statement, Implementation Status, Technical Requirements, Expected Deliverables, Testing Strategy
    - Check any other sections present in the spec
    - Don't skip any sections, even if they seem less important
    - Identify which sections need updates based on proposed changes
    - Consider how changes might affect other sections
  </mandatory_sections_check>
</comprehensive_analysis>

<analysis_requirements>
  <complete_spec_read>
    - Read the entire spec file from beginning to end
    - Map all sections and their current content
    - Identify sections that need updates
    - Identify sections that might be affected by changes
    - Don't assume only requested sections need updates
  </complete_spec_read>
  <testing_procedures_check>
    - Check if spec includes testing procedures
    - Verify testing procedures follow appropriate protocols
    - Identify any missing testing requirements
    - Note if testing procedures need updates
    - Verify spec points to tasks.md for implementation details
    - Check for any duplicated implementation details between spec and tasks.md
  </testing_procedures_check>
</analysis_requirements>

<output_requirements>
  - Complete spec analysis summary
  - ALL sections and their completeness
  - ALL sections that need updates
  - ALL sections that might be affected by changes
  - Testing procedures status and requirements
  - Recommendations for comprehensive updates
</output_requirements>

<instructions>
  ACTION: Use context-fetcher subagent to read complete spec
  ANALYZE: All sections systematically
  IDENTIFY: Sections needing updates and potential impacts
  HIGHLIGHT: Testing procedures status and any missing requirements
</instructions>

</step>

<step number="3" name="function_analysis">

### Step 3: Function Analysis & Context

Perform function analysis to understand current codebase state and assess impact of proposed changes on existing functionality.

<analysis_requirements>
  <dependency_analysis>
    - Review current architecture and module organization
    - Identify potential integration points for updated functionality
    - Check for existing patterns that should be followed
    - Assess impact of changes on existing code
  </dependency_analysis>
  <existing_functionality_check>
    - Check for existing functions that could handle the spec requirements
    - Identify functions that could be extended rather than creating new ones
    - Assess whether existing functions can be modified to meet spec needs
    - Document any existing functionality that could be leveraged
  </existing_functionality_check>
  <complexity_threshold_assessment>
    - Evaluate complexity of proposed changes against existing functions
    - Determine if changes exceed complexity threshold for function modification
    - If complexity is low: recommend extending existing functions
    - If complexity is high: recommend creating new functions
    - Document rationale for function creation vs. modification approach
  </complexity_threshold_assessment>
  <context_gathering>
    - Identify existing functions that spec changes might replace or extend
    - Document current patterns and conventions to maintain consistency
    - Check for existing utilities or helpers that could be reused
    - Assess technical debt or refactoring opportunities
  </context_gathering>
</analysis_requirements>

<execution_instructions>
  ANALYZE: Current codebase state relevant to spec changes
  CHECK: Existing functionality that could handle updated spec requirements
  EVALUATE: Complexity threshold for function creation vs. modification
  DOCUMENT: Existing functions and patterns that inform spec updates
  IDENTIFY: Integration points and potential conflicts
</execution_instructions>

<output_requirements>
  - Summary of relevant existing functions
  - Existing functionality assessment and recommendations
  - Complexity threshold evaluation and function approach rationale
  - Integration recommendations for spec updates
</output_requirements>

</step>

<step number="4" name="ai_tool_planning">

### Step 4: AI Tool Planning (MANDATORY)

**MANDATORY REQUIREMENT: After determining the problem, ALWAYS use sequential-thinking and context7 to plan the fix and ensure most updated information.**

<sequential_thinking_requirement>
  **MANDATORY**: Use sequential-thinking MCP to:
  - Break down the update problem into logical steps
  - Identify potential risks and dependencies
  - Plan the optimal approach for spec updates
  - Consider impact on related files and systems
  - Outline success criteria and validation steps
</sequential_thinking_requirement>

<context7_requirement>
  **MANDATORY**: Use Context7 MCP to:
  - Fetch latest documentation for any libraries/frameworks mentioned
  - Get current best practices for testing approaches
  - Ensure technical requirements are based on current standards
  - Verify any API or tool usage is up-to-date
  - Research optimal solutions for identified problems
</context7_requirement>

<planning_output>
  - Sequential-thinking analysis of the update problem
  - Context7 research results for current best practices
  - Comprehensive update plan with step-by-step approach
  - Risk assessment and mitigation strategies
  - Success criteria and validation methods
</planning_output>

<instructions>
  **ACTION**: ALWAYS use sequential-thinking MCP for problem analysis and planning
  **ACTION**: ALWAYS use Context7 MCP for current documentation and best practices
  **REQUIREMENT**: No spec updates proceed without both AI tool outputs
  **OUTPUT**: Comprehensive plan based on AI tool analysis
</instructions>

</step>

<step number="5" name="update_planning">

### Step 5: Update Planning

Plan the comprehensive updates needed across all spec sections based on analysis and user requirements.

<planning_requirements>
  <section_updates>
    - Overview section updates
    - User stories modifications
    - Spec scope adjustments
    - Technical requirements updates
    - Testing strategy modifications
    - Any other section updates needed
  </section_updates>
  <cross_section_consistency>
    - Ensure all sections remain consistent
    - Update related sections when one section changes
    - Maintain logical flow between sections
    - Verify testing procedures align with technical requirements
  </cross_section_consistency>
  <implementation_impact>
    - Assess impact on existing tasks.md
    - Determine if new tasks need to be added
    - Identify if existing tasks need modification
    - Consider if tasks.md needs complete revision
  </implementation_impact>
</planning_requirements>

<instructions>
  ACTION: Plan comprehensive updates across all spec sections
  ENSURE: Cross-section consistency and logical flow
  ASSESS: Impact on existing tasks and implementation
  PREPARE: Detailed update plan for user review
</instructions>

</step>



<step number="6" subagent="file-creator" name="spec_updates">

### Step 6: Spec Updates

Use the file-creator subagent to update the spec file with all identified changes while preserving existing work.

<update_process>
  <preserve_existing>
    - Keep all existing content that doesn't need changes
    - Update only sections that require modifications
    - Maintain existing structure and formatting
    - Preserve any user-specific content or notes
  </preserve_existing>
  <comprehensive_updates>
    - Update all identified sections
    - Ensure cross-section consistency
    - Maintain logical flow and readability
    - Update testing procedures as needed
  </comprehensive_updates>
</update_process>

<instructions>
  ACTION: Use file-creator subagent to update spec file
  PRESERVE: All existing content that doesn't need changes
  UPDATE: All identified sections comprehensively
  MAINTAIN: Consistency and logical flow
</instructions>

</step>

<step number="7" subagent="file-creator" name="tasks_update">

### Step 7: Tasks Update (MANDATORY)

**ALWAYS update tasks.md to reflect the updated spec, regardless of the nature of changes.**

<mandatory_update_logic>
  **MANDATORY**: Update tasks.md to reflect ALL spec changes
  **NO EXCEPTIONS**: Every spec update requires corresponding tasks.md update
  **PURPOSE**: Ensure implementation tasks stay synchronized with spec requirements
</mandatory_update_logic>

<update_requirements>
  <task_modifications>
    - **MANDATORY**: Add new tasks for any new functionality
    - **MANDATORY**: Modify existing tasks if requirements changed
    - **MANDATORY**: Remove obsolete tasks if functionality removed
    - **MANDATORY**: Update task dependencies if needed
    - **MANDATORY**: Ensure tasks.md reflects ALL spec changes
  </task_modifications>
  <testing_updates>
    - **MANDATORY**: Update testing tasks to reflect new requirements
    - **MANDATORY**: Ensure all new functionality has corresponding tests
    - **MANDATORY**: Update test descriptions if needed
    - **MANDATORY**: Align testing approach with updated spec
  </testing_updates>
</update_requirements>

<instructions>
  ACTION: Use file-creator subagent to update tasks.md (MANDATORY)
  MODIFY: Tasks to reflect ALL spec changes
  ENSURE: Testing tasks align with new requirements
  MAINTAIN: Task structure and dependencies
  **REQUIREMENT**: Every spec update MUST include tasks.md update
</instructions>

</step>

<step number="8" name="verification">

### Step 8: Verification

Verify that all updates are complete and consistent across the spec and related files.

  <verification_checklist>
  - [ ] All identified sections updated
  - [ ] Cross-section consistency maintained
  - [ ] Testing procedures updated and aligned
  - [ ] Tasks.md updated (MANDATORY)
  - [ ] No broken references or links
  - [ ] Spec remains clear and readable
  - [ ] All changes preserve existing work
</verification_checklist>

<instructions>
  ACTION: Verify all updates are complete and consistent
  CHECK: Cross-section consistency and testing alignment
  CONFIRM: No broken references or readability issues
</instructions>

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
    - [ ] Spec identified and analyzed comprehensively
    - [ ] All sections checked for updates needed
    - [ ] Function analysis completed
    - [ ] AI tool planning completed (sequential-thinking + context7)
    - [ ] Update plan created and executed
    - [ ] Spec updated comprehensively
    - [ ] Tasks.md updated (MANDATORY)
    - [ ] Cross-section consistency maintained
    - [ ] Testing procedures aligned
    - [ ] All changes preserve existing work
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

<todo_list_guidelines>
  **Todo List Guidelines:**
  - **Use for tracking**: Development progress, implementation steps, testing phases
  - **Update promptly**: Mark tasks complete as soon as they're done
  - **Be specific**: Use clear, actionable task descriptions
  - **Group logically**: Organize related tasks together
  - **Include context**: Add relevant details for complex tasks
  - **Include testing**: Every task should include test creation and verification steps following code-style standards
</todo_list_guidelines> 