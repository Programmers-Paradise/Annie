# Changelog Automation

This directory contains scripts and workflows that automatically maintain the project's `CHANGELOG.md` file.

## Overview

The changelog is updated automatically when pull requests are merged into the main branch. The system:

- ✅ **Categorizes changes** automatically (Added, Changed, Fixed, Security, etc.)
- ✅ **Extracts metadata** from PR titles, bodies, and labels
- ✅ **Maintains history** of all merged PRs
- ✅ **Supports manual population** from historical PR data
- ✅ **Uses GitHub API** when available, with git fallback

## How It Works

### Automatic Updates (GitHub Actions)

When a PR is merged, the `update-changelog.yml` workflow:

1. **Triggers** on PR close event (`closed`)
2. **Verifies** the PR was merged (not just closed)
3. **Extracts** PR metadata (title, body, labels)
4. **Categorizes** the change based on labels and keywords
5. **Updates** `.github/CHANGELOG.md`
6. **Commits** changes back to the repository

### Category Detection

The system automatically assigns changes to categories based on:

**Labels** (highest priority):
- `feature`, `enhancement`, `added`, `add`, `new` → **Added**
- `bug`, `bugfix`, `fix`, `fixed` → **Fixed**
- `breaking`, `breaking-change`, `changed`, `change` → **Changed**
- `deprecated` → **Deprecated**
- `removed` → **Removed**
- `security` → **Security**

**Keywords** in PR title/body (case-insensitive):
- "feature", "enhancement", "add", "new" → **Added**
- "fix", "bug" → **Fixed**
- "change", "update", "refactor" → **Changed**
- "remove", "delete" → **Removed**
- "security" → **Security**

**Default**: **Changed** (if no label or keyword matches)

### Entry Extraction

The changelog entry for each PR is extracted from (in order):

1. **Explicit `## Changelog` section** in PR body
   ```markdown
   ## Changelog
   - New feature X
   - Performance improvement Y
   ```

2. **PR title** (with cleanup):
   - Removes conventional commit prefixes (`feat:`, `fix:`, etc.)
   - Removes common template text
   - Example: `"feat: Add new search method"` → `"Add new search method"`

3. **Fallback**: PR number with link `"PR #{number} (URL)"`

## Usage

### Automatic (Recommended)

Just merge your PRs normally. The workflow runs automatically on merge and updates the changelog.

### Manual Population (Historical Data)

To populate the changelog with historical PR data:

#### Option 1: Using the Script Locally

```bash
# Install dependencies
pip install requests

# Populate from all merged PRs
python .github/scripts/populate_changelog.py

# With custom limit
python .github/scripts/populate_changelog.py --limit 50

# Since specific date
python .github/scripts/populate_changelog.py --since 2025-06-01
```

#### Option 2: Using GitHub Actions Dispatch

Go to **Actions** → **Update Changelog on PR Merge** → **Run workflow**

Set `Build changelog from historical PR data` to `true`

### Authentication (Optional)

For private repositories or higher API rate limits, set a GitHub token:

```bash
export GITHUB_TOKEN="ghp_YOUR_TOKEN_HERE"
python .github/scripts/populate_changelog.py
```

Without a token, the script falls back to parsing git merge commits locally.

## Configuration

### Workflow Files

- **`.github/workflows/update-changelog.yml`**: Main GitHub Actions workflow
  - Triggers: PR merge events
  - Can be manually triggered for historical population
  
### Scripts

- **`.github/scripts/update_changelog.py`**: Core logic
  - PR metadata extraction
  - Category detection
  - Changelog formatting
  - Supports both API and git fallback
  
- **`.github/scripts/populate_changelog.py`**: Utility for batch population
  - Fetches historical PR data
  - Populates changelog in bulk
  - CLI interface with options

## Changelog Format

Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/):

```markdown
# Changelog

## [Unreleased]

### Added
- Feature X ([#123](link))

### Fixed
- Bug Y ([#456](link))

## [Version] - YYYY-MM-DD

### Added
- ...
```

### Categories (in order)

1. **Added** - New features
2. **Changed** - Changes to existing functionality
3. **Deprecated** - Soon-to-be removed features
4. **Removed** - Now-removed features
5. **Fixed** - Bug fixes
6. **Security** - Security patches

## Best Practices for PRs

To get the best changelog entries:

1. **Use descriptive PR titles**:
   - ✅ `"Add support for Manhattan distance"`
   - ❌ `"Update code"`

2. **Add relevant labels**:
   - Feature PRs: label as `feature` or `enhancement`
   - Bug fixes: label as `bug` or `fix`
   - Breaking changes: label as `breaking-change`

3. **Include explicit changelog section** (optional):
   ```markdown
   ## Changelog
   
   - Add new `remove()` method for deleting index entries
   - Improve performance with batch insertions
   ```

## Troubleshooting

### Changelog not updating after PR merge

1. Check workflow status in Actions tab
2. Verify PR was merged (not just closed)
3. Check PR title/body for metadata

### Incorrect category assignment

1. Add appropriate label to the PR
2. Or include keywords in the title/body
3. Manually edit the changelog if needed

### Rate limiting (GitHub API)

- Without token: 60 requests/hour per IP
- With token: 5000 requests/hour
- Fallback to git parsing works without limit

## Examples

### Before Automation
Manually maintaining changelog entries, risk of:
- Missing updates
- Inconsistent formatting  
- Outdated information

### After Automation
All merged PRs automatically tracked:
```
## [Unreleased]

### Added
- Add support for Manhattan (L1) distance ([#123](link))
- Implement `remove()` method for entry deletion ([#124](link))

### Fixed
- Fix index corruption on large datasets ([#125](link))

### Changed
- Update documentation for new APIs ([#126](link))
```

## Future Enhancements

Potential improvements:

- [ ] Automatic version bumping based on change types
- [ ] Release notes generation from changelog
- [ ] Changelog validation in PR checks
- [ ] Custom category mapping per project
- [ ] Multiple changelog files (by component)
- [ ] Changelog preview in PR comments

## Support

For issues or questions about the changelog automation, open an issue or check the workflow logs in GitHub Actions.
