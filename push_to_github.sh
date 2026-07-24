#!/usr/bin/env bash
# Push reviewed commits with one encrypted, repository-scoped GitHub SSH key.
# The key passphrase is never stored, and any inherited ssh-agent is disabled.

set -Eeuo pipefail
IFS=$'\n\t'

readonly PROGRAM="${0##*/}"
readonly DEFAULT_REPOSITORY="SWT110/eeg"

: "${HOME:?HOME is not set}"

repository="${GITHUB_REPOSITORY:-$DEFAULT_REPOSITORY}"
key_file="${GITHUB_DEPLOY_KEY:-$HOME/.ssh/github_eeg_deploy}"
known_hosts="${GITHUB_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
target_branch=""
allow_dirty=0
dry_run=0
assume_yes=0

usage() {
    cat <<EOF
Usage: $PROGRAM [options]

Push the current committed HEAD to GitHub using an encrypted SSH deploy key.
This script never stages or commits files and never stores the key passphrase.

Options:
  --key PATH          Encrypted private key
                      (default: \$HOME/.ssh/github_eeg_deploy)
  --repo OWNER/REPO   GitHub destination (default: $DEFAULT_REPOSITORY)
  --branch NAME       Destination branch (default: current branch)
  --known-hosts PATH  SSH known_hosts file (default: \$HOME/.ssh/known_hosts)
  --allow-dirty       Push committed HEAD even if the worktree is dirty
  --dry-run           Ask GitHub to check the push without updating refs
  -y, --yes           Skip the destination confirmation
  -h, --help          Show this help

Environment equivalents:
  GITHUB_DEPLOY_KEY, GITHUB_REPOSITORY, GITHUB_KNOWN_HOSTS

Security properties:
  * only SSH public-key authentication is allowed;
  * inherited ssh-agent and askpass helpers are disabled;
  * the private key must be owned by this user, have no group/other access,
    and be protected by a non-empty passphrase;
  * GitHub host-key checking is strict;
  * no force push, automatic staging, or automatic commit is performed.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --key)
            (($# >= 2)) || die "--key requires a path"
            key_file=$2
            shift 2
            ;;
        --repo)
            (($# >= 2)) || die "--repo requires OWNER/REPO"
            repository=$2
            shift 2
            ;;
        --branch)
            (($# >= 2)) || die "--branch requires a branch name"
            target_branch=$2
            shift 2
            ;;
        --known-hosts)
            (($# >= 2)) || die "--known-hosts requires a path"
            known_hosts=$2
            shift 2
            ;;
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -y|--yes)
            assume_yes=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 0)) || die "unexpected positional arguments: $*"
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] ||
    die "invalid GitHub repository '$repository'; expected OWNER/REPO"

for command_name in git ssh ssh-keygen stat mktemp readlink; do
    command -v "$command_name" >/dev/null 2>&1 ||
        die "required command not found: $command_name"
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" ||
    die "the script is not inside a Git repository"
cd -- "$repo_root"

git rev-parse --verify HEAD >/dev/null 2>&1 || die "the repository has no commits"
source_branch="$(git symbolic-ref --quiet --short HEAD)" ||
    die "detached HEAD is not supported"
if [[ -z "$target_branch" ]]; then
    target_branch=$source_branch
fi
git check-ref-format --branch "$target_branch" >/dev/null 2>&1 ||
    die "invalid destination branch: $target_branch"

worktree_status="$(git status --porcelain=v1 --untracked-files=normal)"
if [[ -n "$worktree_status" ]]; then
    printf 'The worktree has uncommitted changes:\n%s\n\n' "$worktree_status" >&2
    if ((allow_dirty == 0)); then
        die "review and commit/ignore these files first, or use --allow-dirty to push committed HEAD only"
    fi
    printf 'WARNING: only committed HEAD will be pushed; the changes above are not included.\n\n' >&2
fi

[[ -f "$key_file" ]] || die "private key not found: $key_file"
[[ -r "$key_file" ]] || die "private key is not readable: $key_file"
key_file="$(readlink -f -- "$key_file")"
[[ -O "$key_file" ]] || die "private key must be owned by the current user: $key_file"
[[ "$key_file" != "$repo_root" && "$key_file" != "$repo_root/"* ]] ||
    die "private key must be stored outside the Git repository: $key_file"

key_mode="$(stat -c '%a' -- "$key_file")"
if (( (8#$key_mode & 8#077) != 0 )); then
    die "private key permissions are $key_mode; run: chmod 600 '$key_file'"
fi

# With an empty passphrase, -y succeeds. Refuse such keys. For an encrypted key
# it fails without opening an interactive prompt, and ssh asks for the real
# passphrase later during this one push.
if ssh-keygen -y -P '' -f "$key_file" >/dev/null 2>&1; then
    die "the private key has no passphrase; create an encrypted deploy key"
fi

[[ -f "$known_hosts" ]] || die "known_hosts file not found: $known_hosts"
[[ -r "$known_hosts" ]] || die "known_hosts is not readable: $known_hosts"
known_hosts="$(readlink -f -- "$known_hosts")"
ssh-keygen -F github.com -f "$known_hosts" >/dev/null 2>&1 ||
    die "github.com is absent from $known_hosts; verify and add GitHub's official SSH host key first"

push_url="ssh://git@github.com/${repository}.git"
commit_id="$(git rev-parse --short=12 HEAD)"
commit_subject="$(git log -1 --format=%s HEAD)"

printf 'Repository : %s\n' "$push_url"
printf 'Source     : %s (committed HEAD only)\n' "$source_branch"
printf 'Destination: %s\n' "$target_branch"
printf 'Commit     : %s %s\n' "$commit_id" "$commit_subject"
((dry_run == 0)) || printf 'Mode       : dry-run\n'

if ((assume_yes == 0)); then
    [[ -t 0 && -t 1 ]] || die "confirmation requires a terminal; use --yes for non-interactive invocation"
    read -r -p 'Continue and enter the SSH key passphrase? [y/N] ' reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || {
        printf 'Cancelled.\n'
        exit 0
    }
fi

# GIT_SSH does not accept options itself, so use a private temporary wrapper.
# It explicitly ignores ~/.ssh/config, inherited agents, and askpass programs.
tmp_parent="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}"
[[ -d "$tmp_parent" && -w "$tmp_parent" ]] || tmp_parent=/tmp
tmp_dir="$(umask 077; mktemp -d -- "$tmp_parent/eeg-github-push.XXXXXXXX")"
cleanup() {
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

ssh_wrapper="$tmp_dir/ssh"
cat >"$ssh_wrapper" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
unset SSH_AUTH_SOCK SSH_AGENT_PID SSH_ASKPASS SSH_ASKPASS_REQUIRE
exec "${GITHUB_PUSH_SSH_BIN:?}" \
    -F /dev/null \
    -i "${GITHUB_PUSH_KEY_FILE:?}" \
    -o IdentityAgent=none \
    -o IdentitiesOnly=yes \
    -o AddKeysToAgent=no \
    -o ForwardAgent=no \
    -o PreferredAuthentications=publickey \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o StrictHostKeyChecking=yes \
    -o UpdateHostKeys=no \
    -o CheckHostIP=no \
    -o "UserKnownHostsFile=${GITHUB_PUSH_KNOWN_HOSTS:?}" \
    "$@"
EOF
chmod 700 "$ssh_wrapper"

push_args=(push --porcelain)
((dry_run == 0)) || push_args+=(--dry-run)
push_args+=("$push_url" "HEAD:refs/heads/$target_branch")

printf '\nThe passphrase prompt comes from OpenSSH and is not saved.\n'
env \
    -u SSH_AUTH_SOCK \
    -u SSH_AGENT_PID \
    -u SSH_ASKPASS \
    -u SSH_ASKPASS_REQUIRE \
    -u GIT_ASKPASS \
    -u GIT_SSH_COMMAND \
    GIT_SSH="$ssh_wrapper" \
    GIT_SSH_VARIANT=ssh \
    GIT_TERMINAL_PROMPT=1 \
    GITHUB_PUSH_SSH_BIN="$(command -v ssh)" \
    GITHUB_PUSH_KEY_FILE="$key_file" \
    GITHUB_PUSH_KNOWN_HOSTS="$known_hosts" \
    git "${push_args[@]}"

printf '\nPush finished. The SSH process has exited; no key was left in an ssh-agent.\n'
