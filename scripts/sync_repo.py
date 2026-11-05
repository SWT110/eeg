"""Synchronize reviewed Git commits between this PC, GitHub and the EEG server.

Never stages/commits, force-pushes, resets, stashes, removes local artifacts or
starts GPU work. Pulls are fast-forward-only and require a clean source tree.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = "/public/home/gd_110413/SWT/work/eeg"


class SyncError(RuntimeError):
    pass


def git(root, *args, capture=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(["git", "-C", str(root), *args], env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail = result.stderr.strip() if capture else "see Git output above"
        raise SyncError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip() if capture else ""


def require_clean(root, branch):
    current = git(root, "symbolic-ref", "--quiet", "--short", "HEAD", capture=True)
    if current != branch:
        raise SyncError(f"Current branch is {current}; expected {branch}. Switch branches explicitly first.")
    changes = git(root, "status", "--porcelain=v1", "--untracked-files=normal", capture=True)
    if changes:
        raise SyncError("Uncommitted source/doc changes exist. Review and commit them before syncing:\n" + changes)


def update(root, remote, branch):
    require_clean(root, branch)
    git(root, "fetch", remote)
    git(root, "merge", "--ff-only", f"{remote}/{branch}")


def push(root, remote, branch):
    update(root, remote, branch)
    git(root, "push", remote, f"HEAD:refs/heads/{branch}")


def deploy(host, branch):
    if host not in {"school-eeg", "school-eeg-2"}:
        raise SyncError("Use one of the configured EEG SSH aliases")
    command = shlex.join(["python3", SERVER_ROOT + "/scripts/sync_repo.py", "pull", "--branch", branch])
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command])
    if result.returncode:
        raise SyncError("Server update failed. PC/GitHub commits remain saved; resolve server changes/connectivity and retry.")


def status(root, remote, branch, refresh=False):
    if refresh:
        git(root, "fetch", remote)
    head = git(root, "rev-parse", "--short=12", "HEAD", capture=True)
    upstream = git(root, "rev-parse", "--short=12", f"{remote}/{branch}", capture=True)
    print(f"PC/current checkout: {head}\nCached {remote}/{branch}: {upstream}")
    git(root, "-c", "core.quotePath=false", "status", "--short", "--branch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "pull", "push", "deploy", "sync"])
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--host", choices=["school-eeg", "school-eeg-2"], default="school-eeg")
    parser.add_argument("--fetch", action="store_true", help="Refresh upstream refs for status")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.remote):
        parser.error("--remote must name a configured remote")
    try:
        git(root, "check-ref-format", "--branch", args.branch, capture=True)
        if args.action == "status":
            status(root, args.remote, args.branch, args.fetch)
        elif args.action == "pull":
            update(root, args.remote, args.branch)
            status(root, args.remote, args.branch)
        elif args.action == "push":
            push(root, args.remote, args.branch)
            status(root, args.remote, args.branch)
        elif args.action == "deploy":
            require_clean(root, args.branch)
            deploy(args.host, args.branch)
        else:
            push(root, args.remote, args.branch)
            deploy(args.host, args.branch)
            status(root, args.remote, args.branch)
    except SyncError as error:
        print(f"Sync stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
