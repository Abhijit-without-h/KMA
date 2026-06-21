# Releasing KMA to PyPI

Distribution name: **`kleinian-memory`** (import as `kma`). The name is currently
free on PyPI.

## 1. Build & validate

```bash
python -m build                 # -> dist/*.whl and dist/*.tar.gz
python -m twine check dist/*    # must print PASSED for both
```

Sanity-check the wheel in a clean environment:

```bash
python -m venv /tmp/kv && /tmp/kv/bin/pip install dist/*.whl
/tmp/kv/bin/python -c "import kma; from kma.memory import AgenticMemory; AgenticMemory().add('hi'); print('ok')"
```

## 2. Publish — pick one

### Option A — Trusted Publishing (recommended, no tokens)

The repo already ships `.github/workflows/publish.yml`, which publishes on a
`v*` tag via PyPI OIDC.

1. On PyPI → *Account → Publishing → Add a pending publisher*:
   - PyPI project name: `kleinian-memory`
   - Owner: `Abhijit-without-h`  ·  Repo: `KMA`
   - Workflow: `publish.yml`  ·  Environment: `pypi`
2. In GitHub repo settings → *Environments* → create an environment named `pypi`.
3. Tag and push:

```bash
git tag v0.1.0
git push origin v0.1.0          # fires the publish workflow
```

### Option B — Manual upload (API token)

```bash
# create a token at https://pypi.org/manage/account/token/
python -m twine upload dist/*   # username: __token__   password: <pypi-token>
```

## 3. Each subsequent release

- Bump `version` in `pyproject.toml` (PyPI rejects re-uploading an existing version).
- Rebuild, `twine check`, then tag `vX.Y.Z` (Option A) or `twine upload` (Option B).

## Notes

- **License:** AGPL-3.0-or-later. This is strong copyleft — downstream users
  (including networked/SaaS use, per AGPL §13) must keep your credit and release
  source under the same license. Intended, but be aware it deters some commercial
  adoption.
- **Optional extras** users can install: `kleinian-memory[embed]` (sentence-
  transformers), `[train]` (torch), `[mcp]` (MCP server), `[viz]` (matplotlib).
- The MCP server runs as `kma-mcp` after install (console entry point).
