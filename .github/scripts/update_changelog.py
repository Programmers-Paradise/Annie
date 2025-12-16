#!/usr/bin/env python3
"""
Update CHANGELOG.md automatically when PRs are merged.
Extracts metadata from PR title, body, and labels to categorize changes.
"""

import os
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import subprocess

try:
    import requests
except ImportError:
    print("Installing requests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"], timeout=30)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requests: {e}")
        sys.exit(1)
    import requests


CHANGELOG_PATH = Path(".github/CHANGELOG.md")
GITHUB_API = "https://api.github.com"

# Parse repository from environment or git config
_repo = os.getenv("GITHUB_REPOSITORY", "")
if "/" in _repo:
    REPO_OWNER, REPO_NAME = _repo.split("/", 1)
else:
    # Fallback to git config - lazy-loaded only if needed
    REPO_OWNER, REPO_NAME = None, None

    def _get_repo_from_git():
        """Lazily fetch repo owner/name from git config (cached after first call)."""
        global REPO_OWNER, REPO_NAME
        if REPO_OWNER is not None:
            return REPO_OWNER, REPO_NAME

        import subprocess

        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], text=True, timeout=5
            ).strip()
            # Strictly validate remote_url before parsing
            if not re.match(
                r"^(git@github\.com:[^/]+/.+?\.git|https://github\.com/[^/]+/.+?(/|\.git)?)$",
                remote_url,
            ):
                raise ValueError(f"Untrusted or malformed remote URL: {remote_url}")
            # Use regex to safely extract owner/repo from known GitHub URL patterns
            ssh_match = re.match(
                r"git@github\.com:([^/]+)/(.+?)(?:\.git)?$", remote_url
            )
            https_match = re.match(
                r"https://github\.com/([^/]+)/(.+?)(?:\.git)?/?$", remote_url
            )

            if ssh_match:
                REPO_OWNER, REPO_NAME = ssh_match.groups()
                REPO_NAME = REPO_NAME.rstrip("/")
            elif https_match:
                REPO_OWNER, REPO_NAME = https_match.groups()
                REPO_NAME = REPO_NAME.rstrip("/")
            else:
                REPO_OWNER, REPO_NAME = "unknown", "unknown"
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            REPO_OWNER, REPO_NAME = "unknown", "unknown"

        return REPO_OWNER, REPO_NAME


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Category mapping from labels/keywords
CATEGORY_MAPPING = {
    "feature": "Added",
    "enhancement": "Added",
    "added": "Added",
    "add": "Added",
    "new": "Added",
    "bug": "Fixed",
    "bugfix": "Fixed",
    "fix": "Fixed",
    "fixed": "Fixed",
    "breaking": "Changed",
    "breaking-change": "Changed",
    "changed": "Changed",
    "change": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "security": "Security",
}

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def get_pr_labels(pr_number: int) -> List[str]:
    """Fetch PR labels from GitHub API."""
    try:
        owner, repo = (
            _get_repo_from_git() if REPO_OWNER is None else (REPO_OWNER, REPO_NAME)
        )
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        pr_data = response.json()
        return [label["name"].lower() for label in pr_data.get("labels", [])]
    except Exception as e:
        print(f"Warning: Could not fetch PR labels: {e}")
        return []


def categorize_change(pr_title: str, pr_body: str, labels: List[str]) -> str:
    """Determine the category (Added, Fixed, Changed, etc.) for a PR."""
    # Check labels first
    for label in labels:
        if label in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[label]

    # Check title and body for keywords
    combined = f"{pr_title} {pr_body}".lower()
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in combined:
            return category

    # Default to "Changed"
    return "Changed"


def extract_changelog_entry(
    pr_title: str, pr_body: str, pr_number: int, pr_url: str
) -> str:
    """
    Extract a changelog entry from PR title and body.
    Looks for:
    1. Explicit changelog section in PR body
    2. Uses PR title as fallback
    """
    # Look for explicit changelog section in PR body
    if pr_body:
        # Pattern: ## Changelog or ### Changelog followed by content
        changelog_match = re.search(
            r"#+\s*changelog[:\s]*\n(.*?)(?:\n#+\s*|$)",
            pr_body,
            re.IGNORECASE | re.DOTALL,
        )
        if changelog_match:
            content = changelog_match.group(1).strip()
            if content:
                return f"{content} ([#{pr_number}]({pr_url}))"

    # Use PR title as fallback, clean it up
    title = pr_title.strip()
    # Remove conventional commit prefixes (feat:, fix:, etc.)
    title = re.sub(
        r"^(feat|fix|docs|style|refactor|perf|test|chore):\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    # Remove PR template remnants
    title = re.sub(r"^\*\*.*?\*\*\s*", "", title)
    title = title.strip()

    if title:
        return f"{title} ([#{pr_number}]({pr_url}))"

    return f"PR #{pr_number} ({pr_url})"


def get_merged_prs_since(
    since_date: Optional[str] = None, limit: int = 100
) -> List[Dict]:
    """Fetch all merged PRs from GitHub API."""
    try:
        owner, repo = (
            _get_repo_from_git() if REPO_OWNER is None else (REPO_OWNER, REPO_NAME)
        )
        query = f"repo:{owner}/{repo} is:pr is:merged"
        if since_date:
            query += f" merged:>={since_date}"

        url = f"{GITHUB_API}/search/issues"
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": min(limit, 100),
        }

        # Use token if available, but make it optional for public repos
        headers = (
            HEADERS.copy()
            if GITHUB_TOKEN
            else {"Accept": "application/vnd.github.v3+json"}
        )
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"Warning: Could not fetch historical PRs from API: {e}")
        print("Falling back to local git history parsing...")
        return get_merged_prs_from_git(limit=limit)


def get_merged_prs_from_git(limit: int = 100) -> List[Dict]:
    """
    Parse merged PRs from git log.
    Looks for merge commits with pattern: "Merge pull request #N"
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "log", "--all", "--oneline", "--merges", "-n", str(limit)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        prs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Parse: "commit_hash Merge pull request #123 from ..."
            match = re.search(r"#(\d+)", line)
            if match:
                pr_number = int(match.group(1))
                # Extract title (part after the PR reference)
                title_match = re.search(r"#\d+\s+from\s+[\w-]+/([\w-]+)\s+(.*)", line)
                if title_match:
                    title = title_match.group(2).strip()
                else:
                    title = line

                owner, repo = (
                    _get_repo_from_git()
                    if REPO_OWNER is None
                    else (REPO_OWNER, REPO_NAME)
                )
                prs.append(
                    {
                        "number": pr_number,
                        "title": title,
                        "body": "",
                        "html_url": f"https://github.com/{owner}/{repo}/pull/{pr_number}",
                        "labels": [],
                    }
                )

        return prs[:limit]
    except Exception as e:
        print(f"Warning: Could not parse git history: {e}")
        return []


def parse_changelog() -> Tuple[str, List[str], str]:
    """
    Parse existing CHANGELOG.md and return:
    - content before "## [Unreleased]"
    - list of merged PR numbers already in changelog
    - content after "## [Unreleased]" section
    """
    if not CHANGELOG_PATH.exists():
        return "", [], ""

    content = CHANGELOG_PATH.read_text(encoding="utf-8")

    # Find the "## [Unreleased]" section
    unreleased_match = re.search(r"(## \[Unreleased\])", content)
    if not unreleased_match:
        # No unreleased section, append to beginning of actual changelog
        return content, [], ""

    before = content[: unreleased_match.start()]
    after_start = unreleased_match.end()

    # Find the next version header after [Unreleased]
    next_version = re.search(r"\n## \[", content[after_start:])
    if next_version:
        unreleased_content = content[after_start : after_start + next_version.start()]
        after = content[after_start + next_version.start() :]
    else:
        unreleased_content = content[after_start:]
        after = ""

    # Extract PR numbers from existing unreleased section
    pr_numbers = set(re.findall(r"\(#(\d+)\)", unreleased_content))

    return before, list(pr_numbers), after


def format_changelog_section(category: str, entries: List[str]) -> str:
    """Format a changelog category section."""
    if not entries:
        return ""

    result = f"\n### {category}\n"
    for entry in entries:
        result += f"- {entry}\n"
    return result


def build_changelog_content(prs: List[Dict]) -> str:
    """Build the Unreleased section content from merged PRs."""
    entries_by_category: Dict[str, List[str]] = {}

    # Category order
    category_order = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]

    for pr in prs:
        pr_number = pr["number"]
        pr_title = pr["title"]
        pr_body = pr["body"] or ""
        pr_url = pr["html_url"]

        labels = [label["name"].lower() for label in pr.get("labels", [])]
        category = categorize_change(pr_title, pr_body, labels)
        entry = extract_changelog_entry(pr_title, pr_body, pr_number, pr_url)

        if category not in entries_by_category:
            entries_by_category[category] = []
        entries_by_category[category].append(entry)

    # Build content with proper ordering
    content = "## [Unreleased]\n"
    for category in category_order:
        if category in entries_by_category:
            content += format_changelog_section(category, entries_by_category[category])

    return content


def update_changelog_single_pr(
    pr_number: int, pr_title: str, pr_body: str, pr_author: str, pr_url: str
):
    """Update changelog for a single merged PR."""
    before, existing_prs, after = parse_changelog()

    # Skip if PR already in changelog
    if int(pr_number) in map(int, existing_prs):
        print(f"PR #{pr_number} already in changelog")
        return False

    # Fetch labels from GitHub API
    labels = get_pr_labels(pr_number)

    # Determine category and create entry
    category = categorize_change(pr_title, pr_body, labels)
    entry = extract_changelog_entry(pr_title, pr_body, pr_number, pr_url)

    # Parse existing unreleased section
    if before and after:
        # Extract existing unreleased sections
        unreleased_content = (
            before.split("## [Unreleased]")[1] if "## [Unreleased]" in before else ""
        )
        full_before = (
            before.split("## [Unreleased]")[0]
            if "## [Unreleased]" in before
            else before
        )
    else:
        unreleased_content = ""
        full_before = before

    # Parse existing categories
    entries_by_category: Dict[str, List[str]] = {}
    category_order = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]

    # Extract existing entries from unreleased section
    for cat in category_order:
        pattern = rf"### {cat}\n((?:- .*\n)*)"
        match = re.search(pattern, unreleased_content)
        if match:
            entries = [
                line.strip()
                for line in match.group(1).split("\n")
                if line.strip().startswith("-")
            ]
            entries_by_category[cat] = [e[2:] for e in entries]  # Remove "- "

    # Add new entry
    if category not in entries_by_category:
        entries_by_category[category] = []
    entries_by_category[category].insert(0, entry)

    # Rebuild unreleased section
    unreleased_section = "## [Unreleased]\n"
    for category in category_order:
        if category in entries_by_category:
            unreleased_section += format_changelog_section(
                category, entries_by_category[category]
            )

    # Reconstruct full changelog
    new_content = full_before + unreleased_section + after

    CHANGELOG_PATH.write_text(new_content, encoding="utf-8")
    print(f"Updated changelog for PR #{pr_number}")
    return True


def build_full_changelog_from_history(limit: int = 100):
    """Build complete changelog from all merged PRs in repo history."""
    print("Building changelog from historical PR data...")

    # Fetch all merged PRs
    prs = get_merged_prs_since(limit=limit)
    print(f"Found {len(prs)} merged PRs")

    if not prs:
        print("No merged PRs found")
        return

    # Parse current changelog to preserve already-included PRs
    _, existing_prs, _ = parse_changelog()
    existing_pr_set = set(map(int, existing_prs))

    # Filter out already-included PRs
    new_prs = [pr for pr in prs if pr["number"] not in existing_pr_set]
    print(f"Adding {len(new_prs)} new PRs to changelog")

    # Build new changelog content
    new_unreleased = build_changelog_content(new_prs)

    # Read current changelog and replace unreleased section
    if CHANGELOG_PATH.exists():
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        # Replace or add unreleased section
        if "## [Unreleased]" in content:
            # Find the next version after unreleased
            unreleased_pattern = r"## \[Unreleased\].*?(?=\n## \[|\Z)"
            new_content = re.sub(
                unreleased_pattern, new_unreleased, content, flags=re.DOTALL
            )
        else:
            # Insert unreleased section at the top of the actual changelog
            changelog_start = content.find("# Changelog")
            if changelog_start != -1:
                new_content = (
                    content[:changelog_start]
                    + new_unreleased
                    + "\n\n"
                    + content[changelog_start:]
                )
            else:
                new_content = new_unreleased + "\n\n" + content
    else:
        new_content = new_unreleased

    CHANGELOG_PATH.write_text(new_content, encoding="utf-8")
    print("Changelog updated with historical data")


def main():
    """Main entry point."""
    event_action = os.getenv("EVENT_ACTION", "")
    pr_number = os.getenv("PR_NUMBER", "")
    pr_title = os.getenv("PR_TITLE", "")
    pr_body = os.getenv("PR_BODY", "")
    pr_author = os.getenv("PR_AUTHOR", "")
    pr_url = os.getenv("PR_URL", "")
    is_historical = os.getenv("IS_HISTORICAL", "").lower() == "true"

    # Only process if PR was closed (merged)
    if event_action == "closed" and pr_number:
        if update_changelog_single_pr(pr_number, pr_title, pr_body, pr_author, pr_url):
            os.environ["CHANGELOG_UPDATED"] = "true"
    elif is_historical:
        build_full_changelog_from_history()
        os.environ["CHANGELOG_UPDATED"] = "true"
    else:
        print("No action needed")


if __name__ == "__main__":
    main()
