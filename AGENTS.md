# Instructions for coding agents

## Git workflow

- Work on a feature branch; never push directly to `main`.
- Run relevant tests and commit hooks before committing.
- When the user requests a push, commit the completed changes and push the
  current feature branch to `origin`.
- Never force-push unless explicitly requested.

## Pre-commit cache in the managed cloud workspace

The writable pre-commit cache path in this environment is
`/workspace/.cache/pre-commit`. The default cache under
`/home/agent/.cache/pre-commit` may be read-only. Set `PRE_COMMIT_HOME` for
commands that run pre-commit, including `git commit`, which invokes the hooks:

```bash
PRE_COMMIT_HOME=/workspace/.cache/pre-commit pre-commit run --all-files
PRE_COMMIT_HOME=/workspace/.cache/pre-commit git commit -m "Describe the change"
```

If `pre-commit` is not on `PATH`, use the project virtual environment:

```bash
PRE_COMMIT_HOME=/workspace/.cache/pre-commit \
  /workspace/venvs/diffpy-stretched-nmf/bin/python -m pre_commit run --all-files
```

These paths are specific to the managed cloud workspace. On other machines,
use a writable cache and the appropriate project environment.

Hook dependencies may need network access on their first run. Black's worker
processes have also stalled inside this workspace's sandbox; use the supported
permission mechanism when necessary to run the hooks successfully.

Prettier's hook environment previously failed to install because npm rejected
its local Git package. If this recurs, skip only Prettier with
`SKIP=prettier` when no changed files use its supported formats, and run all
applicable hooks. Document any skipped checks in the completion report.
