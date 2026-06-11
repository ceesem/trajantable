# trajan

Reshapable and visualizable connectivity tables

## Team Tools & Domain Context

<!-- See team reference doc for CAVE ecosystem, caveclient, neuroglancer,
     task queue patterns, and connectomics-specific design decisions:
     [link TBD] -->

## Development Environment

```bash
# Install dependencies (creates virtual environment)
uv sync

# Add a new dependency
uv add <package>

# Linting / formatting
uv run ruff check src/
uv run ruff format src/
```

### Key Commands

| Command | Description |
|---------|-------------|
| `poe lab` | Launch Jupyter Lab |
| `poe profile` | Profile CPU with pyinstrument (HTML report) |
| `poe profile-all` | Profile CPU + memory with scalene |
| `poe scratch-lab` | Jupyter Lab in the scratch/ directory |
| `poe test` | Run pytest with coverage |
| `poe doc-preview` | Preview documentation locally |
| `poe drybump patch/minor/major` | Dry-run version bump |
| `poe bump patch/minor/major` | Bump version and create tag |

## Library Development

### Testing Strategy

Implement both integration tests and unit tests:

- **Integration tests**: Real-world workflows, end-to-end functionality
- **Unit tests**: Individual methods, edge cases, error conditions
- **Coverage target**: >90% line coverage, >85% branch coverage

```bash
poe test                                                       # full suite with coverage
uv run pytest tests/test_foo.py -v                             # single file
uv run pytest --cov=trajan --cov-report=html tests/  # HTML report
```

### Release Process

```bash
poe drybump patch   # preview what will change
poe bump patch      # bump version, commit, tag (also: minor, major)
```

This updates `pyproject.toml`, `src/trajan/__init__.py`, commits, and tags.

### Scratch Dir

`poe scratch-lab` opens Jupyter Lab in `scratch/`, isolated from the main package environment.

---

## Project Overview

<!-- Describe what this project does and why it exists -->

## Architecture & Key Files

<!-- List key files and how they fit together -->

## Data & External Dependencies

<!-- Document datasets, services, APIs, and file paths this project depends on -->

## Conventions & Patterns

<!-- Describe coding conventions, naming patterns, and design decisions -->

## Notes

<!-- Miscellaneous notes, gotchas, TODOs -->
