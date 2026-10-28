# CLAUDE.md

A feed-forward neural network built from scratch in Python as a learning project.
It has two goals:

1. Understand the maths, so the implementation is written by hand.
2. Practise strong software engineering. Code should meet the standard of a
   well-run production codebase, not a research notebook.

## Commands

- Install: `uv sync --all-extras --dev`
- Test: `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80`
- Lint and format: `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/`
- Type check: `uv run mypy src/` (strict; runs in pre-commit, not yet in CI)
- Download a dataset: `uv run download-data configs/datasets/mnist.yaml data/MNIST`

## Rules

### Dependencies
- Neural network maths uses NumPy and the standard library only. Never use PyTorch,
  TensorFlow, JAX, scikit-learn, autograd libraries, or any package that implements
  layers, activations, losses, gradients or optimisers.
- Other packages are allowed only for software engineering, not ML: Pydantic for
  config, loguru for logging, PyYAML for config files, matplotlib for plots.
- A new runtime dependency must be added to `pyproject.toml` and justified in the PR.

### Code style
- Every function in `src/` has full type hints and a Google-style docstring with
  `Args:`, `Returns:` and `Raises:` sections. Each argument's type goes in
  parentheses, and each exception the function can raise is listed under `Raises:`.
- Module-internal helpers are prefixed with `_`.
- Use `pathlib.Path`, not `os.path` or string paths.
- Log with `loguru.logger`, never `print`. Log at `warning` (or `error` at the top
  level) before raising.

### Design
- Each function does one thing. If describing it needs "and", split it.
- Keep pure computation separate from I/O. Maths functions take arrays and return
  arrays; they don't read files, use global state, or mutate their inputs unless
  the name and docstring say so.
- Don't duplicate logic. When the same block appears twice, extract a helper.
- Interchangeable components (activations, losses, optimisers, layers) share a
  common interface (`typing.Protocol` or an ABC), so adding a new one doesn't mean
  editing existing code or adding `if`/`elif` chains on a name.
- Pass collaborators in (random generators, paths, config objects) rather than
  creating them inside a function, so the code can be tested without patching.
- Validate inputs at public entry points with clear error messages, rather than
  letting a bad value fail deep inside.
- No magic numbers: use a named `UPPER_CASE` module constant or a config field.
- Names say what something is for. Single letters are fine only for standard maths
  notation (`W`, `b`, `x`, `z`, `i`) that the docstring defines.
- Don't import another module's `_`-prefixed helpers. If you need one, make it public.
- No commented-out code. A `TODO` must reference a ticket, e.g. `TODO(NN-30)`.

### Errors
- Raise the project's exceptions from `neural_network/errors.py`, not bare
  `Exception` or `RuntimeError`. Add a new subclass there when no existing one fits.
- When converting a lower-level exception, chain it: `raise DownloadError from ex`.
- Catch the narrowest exception that applies. Never use a bare `except:` or
  `except Exception`, and never silently swallow an error.
- Scripts in `neural_network/scripts/` catch project exceptions, log with
  `logger.error`, and exit with `sys.exit(1)`. Library code never calls `sys.exit`.

### Config
- Configs live as YAML under `configs/` and are validated by Pydantic models in
  `neural_network/config/`. Load YAML with `yaml.safe_load` only.
- Hard-coded values that someone might want to change belong in config, not in code.

### Numerical code
- Randomness takes an explicit `numpy.random.Generator` (from
  `np.random.default_rng(seed)`). Never call `np.random.seed` or the global
  `np.random.*` functions.
- Watch for numerical stability: subtract the max before `exp` in softmax, clip or
  add an epsilon before `log`, and avoid dividing by values that can be zero.
- State array shapes in the docstring for any function that takes or returns arrays.

### Tests
- Every change to `src/` comes with tests. Coverage must stay at or above 80%.
- `tests/unit/` never touches the network or real datasets. Mock `urlopen` (see
  `tests/unit/test_data.py`) and use `tmp_path` for files.
- Group tests in a class per function under test, named `Test<FunctionName>`.
- Gradient code gets a numerical gradient check (finite differences) in its tests.

### Repository
- Never commit anything under `data/` or `notebooks/`, or model checkpoints.
- Branches are named `<type>/NN-<ticket>/<short-name>`, e.g. `feature/NN-12/activation-functions`.
- Commit messages use Conventional Commits prefixes: `feat`, `fix`, `test`,
  `refactor`, `docs`, `build`, `chore`.
- One logical change per PR.

## Reviewing pull requests

This is a learning project, and the author writes the implementation themself.
They also want to learn industry practice, so review as a senior engineer would.

- Priority order: correctness bugs (including wrong gradients, shape mismatches and
  numerical instability), then missing or weak tests, then design, then style.
- Flag design problems (mixed responsibilities, duplication, tight coupling,
  hard-to-test code, unclear names, missing input validation) and name the
  principle behind each, e.g. single responsibility or dependency injection.
- For neural network maths (forward pass, backpropagation, activations, losses,
  optimisers), explain what is wrong and why, and point to the concept involved.
  Do not supply a full replacement implementation.
- Small concrete fixes are fine for non-ML code (I/O, config, scripts, tests).
