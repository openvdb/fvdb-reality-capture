# AGENTS.md — AI Agent Guidelines for fVDB Reality Capture

This file provides persistent instructions for AI coding agents (Cursor, Copilot,
Codex, etc.) working on this codebase. Human contributors should also refer to
[CONTRIBUTING.md](CONTRIBUTING.md).

## Git Commits

- All commits **must** include a DCO sign-off (`git commit --signoff` / `-s`).
- All commits **must** be SSH-signed (`git commit --gpg-sign` / `-S`).
- **Never** skip hooks or signing: do not use `--no-verify` or `--no-gpg-sign`.

## Python Code Style

- Format Python code with **black** using the exact flags from CI:

  ```
  black --target-version=py311 --line-length=120 .
  ```

- **Do not** run `black` with default settings — the default line-length (88) does
  not match this project's setting (120).
- There is currently no `[tool.black]` section in `pyproject.toml`; a future PR may
  add one so that plain `black .` picks up the correct settings automatically.

## License Headers

Every source file **must** include the Apache-2.0 SPDX identifier.

Python files:

```python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
```

## Whitespace

- No trailing whitespace (CI enforces this; `.wlt` files are excluded).
- Use spaces, not tabs (binary and a few config files are excluded from this check).

## Style

- Avoid excessive use of emoji and non-ascii characters in code and documentation except where it is helpful to user
  experience or user interfaces.

## Testing

- Run relevant tests before pushing:

  ```
  cd tests && pytest -v unit benchmarks/test_benchmark_contract.py benchmarks/test_comparison_benchmark.py
  ```

- Note: tests require a GPU runner with fvdb-core installed.

## Test Coverage

When adding new config options or features:
- Update `tests/benchmarks/contract.py` to validate new fields
- Add tests in `test_comparison_benchmark.py` or appropriate test file
- Run existing contract tests to ensure backwards compatibility

## Opening Issues

- Set appropriate labels (e.g. `bug`, `enhancement`, `documentation`).
- Reference any related issues or PRs.
- For bugs: provide clear reproduction steps, expected vs actual behavior, and
  environment details.
- Set appropriate labels on the issue.
- Add the issue to the fvdb-reality-capture GitHub project

## Opening Pull Requests

- Reference the issue being fixed (e.g. "Fixes #NNN").
- Include a test plan with specific commands to verify the change.
- Ensure all CI checks pass before requesting review (DCO, codestyle, tests).
- Keep PRs focused on a single concern.
- Set appropriate labels on the PR.
- Add the PR to the fvdb-reality-capture GitHub project

**Do not commit unless directed:**
- `.vscode/settings.json` - local IDE settings
- `.cursor/` - local Cursor config
- Test-specific matrices or configs with hardcoded paths
