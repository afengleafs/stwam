#!/usr/bin/env bash
# Compatibility wrapper — real script lives under scripts/
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/train_stwam_libero_ddp_bg.sh" "$@"
