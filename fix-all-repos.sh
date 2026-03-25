#!/bin/bash

# Script to fix GitHub contributions for all repos
# Author: Sarthak Atrish
# Date: March 25, 2026

echo "=========================================="
echo "GitHub Contributions Fixer"
echo "=========================================="
echo ""

# Configuration
CORRECT_NAME="Sarthak Atrish"
CORRECT_EMAIL="atrish07sarthak@gmail.com"
OLD_EMAIL="sarthakatrish@SARTHAKs-MacBook-Air-2.local"

# List of your repos (all your GitHub repos)
REPOS=(
    "Monestary360"
    "MoneyMap"
    "MyFitnessPal-main"
    "Planorix"
    "Portfolio_MacOs-main"
    "ReVault-v2"
    "Nudgr"
    "FitnessApp"
)
# Note: PL-Predictor already fixed manually, excluding backup folder

# Base directory where your repos are located
BASE_DIR="$HOME/Desktop"

echo "Configuration:"
echo "  Name: $CORRECT_NAME"
echo "  Email: $CORRECT_EMAIL"
echo "  Base Directory: $BASE_DIR"
echo ""
echo "Repos to fix: ${#REPOS[@]} (PL-Predictor already fixed)"
for repo in "${REPOS[@]}"; do
    echo "  - $repo"
done
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Process each repo
for repo in "${REPOS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing: $repo"
    echo "=========================================="
    
    REPO_PATH="$BASE_DIR/$repo"
    
    # Check if repo exists
    if [ ! -d "$REPO_PATH" ]; then
        echo "❌ Repo not found: $REPO_PATH"
        echo "   Skipping..."
        continue
    fi
    
    cd "$REPO_PATH" || continue
    
    # Check if it's a git repo
    if [ ! -d ".git" ]; then
        echo "❌ Not a git repository: $REPO_PATH"
        echo "   Skipping..."
        continue
    fi
    
    echo "✓ Found repo at: $REPO_PATH"
    
    # Create backup
    BACKUP_NAME="${repo}-BACKUP-$(date +%Y%m%d-%H%M%S)"
    echo "📦 Creating backup: $BACKUP_NAME"
    cd "$BASE_DIR" || exit
    cp -r "$repo" "$BACKUP_NAME"
    cd "$REPO_PATH" || exit
    
    # Create backup branch on GitHub
    echo "📦 Creating backup branch on GitHub..."
    git branch backup-before-rewrite 2>/dev/null
    git push origin backup-before-rewrite 2>/dev/null
    
    # Rewrite history
    echo "🔧 Rewriting commit history..."
    FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter "
        OLD_EMAIL='$OLD_EMAIL'
        CORRECT_NAME='$CORRECT_NAME'
        CORRECT_EMAIL='$CORRECT_EMAIL'
        
        if [ \"\$GIT_COMMITTER_EMAIL\" = \"\$OLD_EMAIL\" ]
        then
            export GIT_COMMITTER_NAME=\"\$CORRECT_NAME\"
            export GIT_COMMITTER_EMAIL=\"\$CORRECT_EMAIL\"
        fi
        if [ \"\$GIT_AUTHOR_EMAIL\" = \"\$OLD_EMAIL\" ]
        then
            export GIT_AUTHOR_NAME=\"\$CORRECT_NAME\"
            export GIT_AUTHOR_EMAIL=\"\$CORRECT_EMAIL\"
        fi
    " --tag-name-filter cat -- --branches --tags 2>&1 | grep -v "^Rewrite"
    
    # Force push
    echo "⬆️  Pushing to GitHub..."
    git push --force origin main 2>&1 | tail -3
    
    # Clean up
    echo "🧹 Cleaning up..."
    rm -rf .git/refs/original/
    git reflog expire --expire=now --all 2>/dev/null
    git gc --prune=now --aggressive 2>&1 | tail -2
    
    echo "✅ Done: $repo"
done

echo ""
echo "=========================================="
echo "✅ All repos processed!"
echo "=========================================="
echo ""
echo "Backups created in: $BASE_DIR"
echo "Backup branches created on GitHub: backup-before-rewrite"
echo ""
echo "Check your contributions: https://github.com/atrishSarthak"
echo ""
echo "To delete backups later:"
echo "  cd $BASE_DIR"
echo "  rm -rf *-BACKUP-*"
echo ""
