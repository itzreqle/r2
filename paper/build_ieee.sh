#!/usr/bin/env bash
# Build paper/r2-ieee.pdf (IEEEtran, ~8 pages incl. references) with tectonic.
#
# references-ieee.bib is generated from references.bib with the project's "Peer-reviewed"
# notes removed (IEEE style shows venues instead). Edit references.bib, not the copy.
# Builds in a temporary directory so no intermediate files land in the repo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v tectonic >/dev/null || { echo "tectonic not found (brew install tectonic)" >&2; exit 1; }
python3 - "$HERE" <<'EOF'
import re, sys, pathlib
here = pathlib.Path(sys.argv[1])
bib = (here / "references.bib").read_text()
bib = re.sub(r"\n\s*note\s*=\s*\{Peer-reviewed[^}]*\},?", "", bib)
(here / "references-ieee.bib").write_text(
    "% Generated from references.bib for r2-ieee.tex (peer-review notes removed). Edit references.bib.\n" + bib)
EOF
OUT="$(mktemp -d -t r2-ieee)"
trap 'rm -rf "$OUT"' EXIT
cp -R "$HERE/r2-ieee.tex" "$HERE/references-ieee.bib" "$HERE/figures" "$OUT/"
cd "$OUT"
tectonic --keep-logs r2-ieee.tex >tectonic.out 2>&1 || { tail -30 tectonic.out >&2; exit 1; }
if grep -qE "undefined references|Reference .* undefined|^!" r2-ieee.log; then
  grep -E "undefined|^!" r2-ieee.log | sort -u >&2
  echo "✗ unresolved references or errors (PDF not copied)" >&2
  exit 1
fi
cp r2-ieee.pdf "$HERE/r2-ieee.pdf"
echo "✓ $HERE/r2-ieee.pdf ($(pdfinfo r2-ieee.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
