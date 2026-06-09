#!/bin/bash
# Sync local changes with remote repository

# Add all changes (including untracked files)
git add -A

# Only commit if there are staged changes
if ! git diff --cached --quiet; then
    git commit -m "Auto-sync: $(date)"
else
    echo "No changes to sync."
fi
