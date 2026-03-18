# How to release

1. Make sure to update README.md and documentations.
2. Run `uv version --bump [major|minor|patch] [--dry-run]`. This will update the `version` field in `pyproject.toml` (equivalent to editing it manually).
3. Run `git-cliff --tag <new-version> --unreleased --prepend CHANGELOG.md`. This will prepend the new release entry to `CHANGELOG.md`.
4. Adjust the latest addition to `CHANGELOG.md` according to [common-changelog](https://common-changelog.org/) standard.
5. Stage `pyproject.toml`, `CHANGELOG.md` and any updated documentations, then commit with `chore(release): v<new-version>`.
6. Create a tag: `git tag v<new-version> -m "release(v<new-version>)[: <brief description>]"`
7. `git push --tags`
8. GitHub release and TestPyPI publishing is automated via `.github/workflows/release.yml`.
