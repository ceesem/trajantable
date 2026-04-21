# TrajanTable

Reshapable and visualizable connectivity tables

## Setup

This project uses `uv` for dependency management and `poe` for task running.

```bash
# Install dependencies
uv sync

# Launch Jupyter Lab
poe lab
```

## Development

### Running Tests

```bash
poe test
```

### Building Documentation

```bash
poe doc-preview
```

### Versioning

```bash
# Dry run to see what will change
poe drybump patch

# Actually bump the version
poe bump patch  # or minor, or major
```



## Profiling

```bash
# Profile with scalene (CPU + memory)
poe profile-all <your script>

# Profile with pyinstrument (CPU only, nicer output)
poe profile <your script>
```

## License

MIT License - see LICENSE file for details.
