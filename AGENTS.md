# Instructions for coding agents

## Git workflow

- Work on a feature branch. Push to `main` only when explicitly requested.
- Run relevant tests and commit hooks before committing.
- When the user requests a push, commit the completed changes and push the
  current feature branch to `origin`.
- Never force-push unless explicitly requested.

When the user explicitly requests pushing specific files to `main`, fetch
the latest `origin/main` and create a separate branch from it. Commit only
the requested files, run the applicable checks, and push that commit with
`git push origin HEAD:refs/heads/main`. This keeps unrelated feature changes
out of `main` and lets the hook that prevents commits on `main` run normally
on the separate branch. If the push is rejected because `main` advanced, fetch and
rebase onto the latest `origin/main`, check the resulting change, and retry
without force-pushing.

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

## Prettier hook compatibility

npm 11.9 rejected Prettier's local Git package during hook environment
installation. Using npm 10 resolved the problem and allowed all applicable
hooks to pass. If this installation error recurs, install npm 10 in the
workspace cache and prepend its executable directory to `PATH` for the
hook or commit command:

```bash
npm install --prefix /workspace/.cache/precommit-npm npm@10
PATH=/workspace/.cache/precommit-npm/node_modules/.bin:$PATH \
  PRE_COMMIT_HOME=/workspace/.cache/pre-commit \
  /workspace/venvs/diffpy-stretched-nmf/bin/python -m pre_commit run --all-files
PATH=/workspace/.cache/precommit-npm/node_modules/.bin:$PATH \
  PRE_COMMIT_HOME=/workspace/.cache/pre-commit \
  git commit -m "Describe the change"
```

Reuse this installation if it already exists. Keep the repository hook
configuration unchanged and run Prettier for Markdown changes.
