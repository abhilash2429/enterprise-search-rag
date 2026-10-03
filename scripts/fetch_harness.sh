#!/usr/bin/env bash
# Sparse clone of the EnterpriseRAG-Bench harness (code only, no corpus) pinned to the commit the spec was written against.
set -euo pipefail
export MSYS_NO_PATHCONV=1
COMMIT=d36685e273713975ee20299bbf1ab64165575b3c
DEST="$(dirname "$0")/../third_party/erb"
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git "$DEST"
  git -C "$DEST" sparse-checkout set --no-cone '/*' '!/generated_data/sources/'
fi
git -C "$DEST" checkout -q "$COMMIT"
echo "harness at $(git -C "$DEST" rev-parse --short HEAD)"
