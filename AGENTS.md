# AGENTS.md

## Pre-Change Inspection
- Inspect the file system and project tree before proposing structural architectural changes.
- Read existing project styles, frameworks, and linting rules; adapt to them instead of introducing new abstractions.
- Keep all modifications tightly scoped to the specific feature or bug requested.
- State your assumptions explicitly when a task description is ambiguous.

## Implementation Standards
- Produce minimal, highly contained diffs to keep code reviews simple.
- Prioritize clear, readable, and idiomatic code over clever or highly nested solutions.
- Never add new npm, pip, or cargo packages unless absolutely critical to the core solution.
- Maintain strict backwards compatibility unless explicitly requested otherwise.

## Code Integrity
- Ensure there are no silent failure catch blocks; bubble errors up properly.
- Use explicit typing and avoid arbitrary fallbacks like `any` or `unknown`.
- Do not fabricate placeholder data inside final production source code.