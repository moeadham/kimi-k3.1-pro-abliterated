#!/usr/bin/env bash
# Push this repo to the dev pod (/root/harness). Pull back with: tools/sync.sh pull <remote-subdir>
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -F tools/ssh_config"
if [ "${1:-push}" = "pull" ]; then
  rsync -az -e "$SSH" "kimi-b200:/root/harness/${2:?subdir}/" "./${2}/"
else
  rsync -az --delete -e "$SSH" \
    --exclude .git --exclude .venv --exclude __pycache__ --exclude tools/.known_hosts \
    --exclude .env --exclude '.env.*' \
    --exclude env.lock.json --exclude 'out/' \
    ./ kimi-b200:/root/harness/
fi
