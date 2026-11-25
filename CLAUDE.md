# Rules for Claude

- Only work on public repos owned by Junaidjameel773. Never read, clone or modify private repos.
- One GitHub issue per PR. Reference it with `Closes #N`.
- Every PR: tests pass (`uv run pytest`), lint clean (`uv run ruff check . && uv run ruff format --check .`).
- Update the Features table and API table in README.md when a feature ships.
- Commits use the repo owner's identity. No AI attribution lines.
- Keep code simple: no speculative abstractions.
