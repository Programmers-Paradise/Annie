#!/usr/bin/env python3
"""
Utility script to populate changelog with historical PR data.
Can be run locally or via GitHub Actions workflow dispatch.

Usage:
    python populate_changelog.py [--limit 100] [--since YYYY-MM-DD]
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from update_changelog import (
    get_merged_prs_since,
    build_changelog_content,
    parse_changelog,
    CHANGELOG_PATH,
)


def populate_changelog(limit: int = 100, since: str = None):
    """Populate changelog with historical PR data."""
    print(
        f"Fetching merged PRs (limit: {limit})" + (f", since {since}" if since else "")
    )

    prs = get_merged_prs_since(since_date=since, limit=limit)
    print(f"Found {len(prs)} merged PRs")

    if not prs:
        print("No PRs found")
        return False

    # Parse current changelog
    before, existing_prs, after = parse_changelog()
    existing_pr_set = set(map(int, existing_prs))

    # Filter new PRs
    new_prs = [pr for pr in prs if pr["number"] not in existing_pr_set]
    print(f"Adding {len(new_prs)} new PRs to changelog")

    if not new_prs:
        print("No new PRs to add")
        return False

    # Build new unreleased section
    new_unreleased = build_changelog_content(new_prs)

    # Read current changelog
    if CHANGELOG_PATH.exists():
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
    else:
        content = "# Changelog\n\nAll notable changes documented here.\n\n"

    # Replace or add unreleased section
    import re

    if "## [Unreleased]" in content:
        # Find the next version header
        unreleased_match = re.search(
            r"## \[Unreleased\].*?(?=\n## \[|\Z)", content, re.DOTALL
        )
        if unreleased_match:
            content = (
                content[: unreleased_match.start()]
                + new_unreleased
                + "\n"
                + content[unreleased_match.end() :]
            )
    else:
        # Insert after "# Changelog" header
        changelog_match = re.search(r"(# Changelog\n)", content)
        if changelog_match:
            insert_pos = changelog_match.end()
            content = (
                content[:insert_pos]
                + "\n"
                + new_unreleased
                + "\n"
                + content[insert_pos:]
            )
        else:
            content = new_unreleased + "\n\n" + content

    CHANGELOG_PATH.write_text(content, encoding="utf-8")
    print(f"✓ Changelog updated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Populate changelog with historical PR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch last 100 merged PRs
  python populate_changelog.py

  # Fetch last 50 PRs
  python populate_changelog.py --limit 50

  # Fetch PRs merged since June 2025
  python populate_changelog.py --since 2025-06-01
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of PRs to fetch (default: 100)",
    )
    parser.add_argument(
        "--since", type=str, help="Only fetch PRs merged since this date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    # Validate date format
    if args.since:
        try:
            datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{args.since}'. Use YYYY-MM-DD")
            sys.exit(1)

    if populate_changelog(limit=args.limit, since=args.since):
        print("\nChangelog population complete!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
