# Releasing MultiClean

How to cut a new release to PyPI. The whole flow is driven from a single
`v*` git tag — `setuptools-scm` reads the version from the tag and
`pypa/gh-action-pypi-publish` ships the wheel.

## Cutting a release

1. **Update [`CHANGELOG.md`](CHANGELOG.md).**
   Promote the `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD` and add
   a fresh empty `[Unreleased]` heading on top. Update the comparison
   links at the bottom of the file to add the new version.

2. **Merge to `main`.**
   Make sure the merge commit is the one you intend to release —
   the tag will be cut from it.

3. **Tag and push.**
   ```bash
   git checkout main
   git pull
   git tag vX.Y.Z              # e.g. v0.3.0
   git push origin vX.Y.Z
   ```

4. **Approve the deployment.**
   The push triggers the [`Publish to PyPI`](.github/workflows/publish.yml)
   workflow. Open *Actions* on GitHub and click *Review pending
   deployments → Approve and deploy* on the `pypi` environment when
   prompted.

5. **Verify on PyPI.**
   Within a minute or two `pip install multiclean==X.Y.Z` should
   work and the project page on
   <https://pypi.org/project/multiclean/> should show the new version.

## Notes

- **Versions come from tags.** `multiclean.__version__` and the wheel
  filename both come from `setuptools-scm` reading the latest `v*` tag.
  Don't hand-edit a version anywhere — bumping a tag is the entire
  bump.
- **Pre-releases** (e.g. `v0.4.0rc1`) work the same way; PEP 440 markers
  in the tag carry through.
- **Yanking a bad release** is done from the PyPI web UI, not from this
  repo. The tag and CHANGELOG entry stay.
- **Hotfix on an older line** (e.g. `v0.3.1` while `main` is on `0.4.x`):
  branch from the older tag, fix, tag `v0.3.1` on that branch, push the
  tag. The same workflow handles it.
