#!/usr/bin/env bash
# Download a HuggingFace model repository for offline use.
# Usage: bash download_model.sh <model_id> [destination]
# Example: bash download_model.sh t5-base models/t5-base

set -euo pipefail

if [[ "$#" -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Download a HuggingFace model repository for offline use." >&2
  echo "Usage: $0 <model_id> [destination_dir]" >&2
  exit 0
fi

MODEL_ID="$1"
DEST="${2:-models/$MODEL_ID}"

echo "Downloading $MODEL_ID to $DEST" >&2

# Ensure git and git-lfs are available
if ! command -v git >/dev/null; then
  echo "Error: git is required but not installed." >&2
  exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
  echo "Error: git-lfs is required but not installed." >&2
  exit 1
fi

git lfs install >/dev/null 2>&1

# Clone the model repository and fetch large files
GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 "https://huggingface.co/${MODEL_ID}" "$DEST"
(
  cd "$DEST"
  git lfs pull --include="*" --exclude="" >/dev/null 2>&1
)

echo "Model saved to $DEST" >&2
