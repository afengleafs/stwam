#!/usr/bin/env bash
# Compatibility wrapper — real script lives under scripts/
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/train_vtwam_libero_ddp_bg.sh" "$@"
