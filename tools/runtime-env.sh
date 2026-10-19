#!/usr/bin/env bash
# Source this file to run the build-time Python tooling (baker, map
# preparation). Dependencies live under .pydeps/ in the repository root so the
# script works both locally and inside CI-like sandboxes with an isolated /tmp.
DIRMAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DIRMAP_ROOT/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
export DIRMAP_CACHE_DIR="${DIRMAP_CACHE_DIR:-$DIRMAP_ROOT/.cache}"
if [ -x "$DIRMAP_ROOT/.ocr/usr/bin/tesseract" ]; then
  export DIRMAP_TESSERACT="$DIRMAP_ROOT/.ocr/usr/bin/tesseract"
  export LD_LIBRARY_PATH="$DIRMAP_ROOT/.ocr/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TESSDATA_PREFIX="$DIRMAP_ROOT/.ocr/usr/share/tesseract-ocr/5/tessdata"
fi
