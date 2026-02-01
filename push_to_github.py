#!/usr/bin/env python3
import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"Error output: {result.stderr}")
            return False
        else:
            print(f"Success: {cmd}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
    except Exception as e:
        print(f"Exception running command: {cmd}")
        print(f"Exception: {e}")
        return False

def main():
    repo_path = "/Users/dhiwahar_k/Desktop/dhiwahar_k/Dhiwahar K Adhithya/Github/Vantage"
    
    print("Starting git operations...")
    
    # Initialize git if not already done
    if not run_command("git init", repo_path):
        print("Failed to initialize git")
        return 1
    
    # Set git configuration
    run_command('git config user.name "DHIWAHAR-K"', repo_path)
    run_command('git config user.email "adhithyak99@gmail.com"', repo_path)
    
    # Add all files
    if not run_command("git add -A", repo_path):
        print("Failed to add files")
        return 1
    
    # Commit with January 31st date
    commit_cmd = 'GIT_AUTHOR_NAME="DHIWAHAR-K" GIT_AUTHOR_EMAIL="adhithyak99@gmail.com" GIT_COMMITTER_NAME="DHIWAHAR-K" GIT_COMMITTER_EMAIL="adhithyak99@gmail.com" git commit -m "Initial commit: Vantage Text-to-SQL project setup" --date="2026-01-31T12:00:00"'
    
    if not run_command(commit_cmd, repo_path):
        print("Failed to create commit")
        return 1
    
    print("Git operations completed successfully!")
    print("Next step: Add GitHub remote and push")
    print("Commands to run manually:")
    print('git remote add origin https://github.com/DHIWAHAR-K/Vantage.git')
    print('git push -u origin main')
    
    return 0

if __name__ == "__main__":
    sys.exit(main())