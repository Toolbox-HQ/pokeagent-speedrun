#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

header() { echo ""; echo "=== $1 ==="; }

header "Repo top-level"
du -sh "$REPO"/*/  2>/dev/null | sort -rh

header ".cache"
du -sh "$REPO"/.cache/*/ 2>/dev/null | sort -rh

header ".cache/pokeagent"
du -sh "$REPO"/.cache/pokeagent/*/ 2>/dev/null | sort -rh

header ".cache/lz"
du -sh "$REPO"/.cache/lz/*/ 2>/dev/null | sort -rh

header "Total"
du -sh "$REPO"
