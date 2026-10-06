# Contributing to open-aiops

Thank you for contributing to OpenAIOps! Please follow these guidelines to ensure smooth collaboration and review.

---

## Workflow & Git Conventions

### 1. Fork and Clone

- Fork the upstream repository: `https://github.com/SupriyankaKharade/open-aiops`
- Clone your fork locally and configure the `upstream` remote:

  ```bash
  git remote add upstream https://github.com/SupriyankaKharade/open-aiops.git
  git fetch upstream
  ```

### 2. Branching Strategy

- Always create your task branch from the latest upstream `main`:

  ```bash
  git switch main
  git merge --ff-only upstream/main
  git push origin main
  git checkout -b task/<task-number>-<short-description>
  ```

- Follow the branch naming convention: `task/<number>-short-slug`
  - Examples: `task/2-core-configuration`, `task/5-model-router-engine`
  - Do **not** use prefixes like `feature/`, `fix/`, or personal names.

### 3. One Task, One Pull Request

- Keep each pull request strictly scoped to one assigned task.
- Do not combine multiple tasks or unrelated refactorings into a single PR.
- Target PRs against `SupriyankaKharade/open-aiops:main`.

### 4. Pull Request Title & Description

- Follow conventional commits for commit messages and PR titles (e.g., `feat(core): ...`, `test(core): ...`, `docs: ...`).
- The **first line** of the PR description should reference the task tracking ID:
  ```markdown
  Closes INT-<number>
  ```
  *(Example: `Closes INT-2`)*
- Include a concise summary of changes and verification instructions.

---

## Code Standards & Hygiene

### 1. Keep Secrets Out of Git

- **Never** commit `.env` files, actual API keys, credentials, or production tokens.
- Use `.env.example` as a template for environment variable documentation.
- Do not commit virtual environments (`.venv`), Python bytecode (`__pycache__`, `*.pyc`), or test caches (`.pytest_cache`).

### 2. Testing Before Submission

- All tests must pass before submitting a pull request:

  ```bash
  python -m pytest -q
  ```

- Include dedicated unit tests under `tests/` for any new functionality or bug fixes.
- Ensure type hints and Pydantic validation are utilized for public interfaces.
