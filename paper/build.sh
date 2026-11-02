#!/usr/bin/env bash
# Build paper/r2-whitepaper.pdf with TinyTeX (pdflatex + bibtex).
#
# One-time setup (what was used for revisions 1 and 2):
#   gh release download v2026.11 -R rstudio/tinytex-releases -p "TinyTeX-1-darwin-v2026.11.tar.xz"
#   tar -xJf TinyTeX-1-darwin-v2026.11.tar.xz -C ~/Library
#   ~/Library/TinyTeX/bin/universal-darwin/tlmgr install newtx pgf listings algorithms \
#       algorithmicx enumitem cleveref multirow caption fontaxes kastrup xstring txfonts
#
# Builds in a temporary directory so no .aux/.log/.bbl files land in the repo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TT="${TINYTEX_BIN:-$HOME/Library/TinyTeX/bin/universal-darwin}"
if [[ ! -x "$TT/pdflatex" ]]; then
  echo "pdflatex not found in $TT (set TINYTEX_BIN or see the setup notes in this script)" >&2
  exit 1
fi
export PATH="$TT:$PATH"
OUT="$(mktemp -d -t r2-paper)"
trap 'rm -rf "$OUT"' EXIT
cp -R "$HERE/r2-whitepaper.tex" "$HERE/references.bib" "$HERE/figures" "$OUT/"
cd "$OUT"
pdflatex -interaction=nonstopmode -halt-on-error r2-whitepaper.tex >/dev/null || {
  grep -E "^!" -A3 r2-whitepaper.log >&2; exit 1; }
bibtex r2-whitepaper >/dev/null
pdflatex -interaction=nonstopmode r2-whitepaper.tex >/dev/null
pdflatex -interaction=nonstopmode r2-whitepaper.tex >/dev/null
if grep -qE "undefined|^!" r2-whitepaper.log; then
  grep -E "undefined|^!" r2-whitepaper.log | sort -u >&2
  echo "✗ unresolved references or errors (PDF not copied)" >&2
  exit 1
fi
cp r2-whitepaper.pdf "$HERE/r2-whitepaper.pdf"
echo "✓ $HERE/r2-whitepaper.pdf ($(pdfinfo r2-whitepaper.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
