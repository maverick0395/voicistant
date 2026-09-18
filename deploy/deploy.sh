#!/usr/bin/env bash
# Deploy on the VPS. Forced command of the CI deploy key (ROADMAP E2), so CI can run this and nothing else.
# CI passes the commit SHA as the ssh "command"; by hand: IMAGE_TAG=<sha> /opt/voicistant/deploy/deploy.sh
set -euo pipefail

tag=${SSH_ORIGINAL_COMMAND:-${IMAGE_TAG:-latest}}
if [[ ! $tag =~ ^([0-9a-f]{40}|latest)$ ]]; then
  echo "refusing tag: $tag" >&2
  exit 1
fi

cd /opt/voicistant
git fetch -q origin main
if [[ $tag == latest ]]; then git checkout -q --detach origin/main; else git checkout -q --detach "$tag"; fi

cd deploy
export IMAGE_TAG=$tag
docker compose --env-file ../.env pull -q
docker compose --env-file ../.env up -d --remove-orphans
docker image prune -f >/dev/null
echo "deployed $tag"
