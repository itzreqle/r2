#!/usr/bin/env bash
# VSCodium installs ms-python.python from Open VSX as a "universal" VSIX that ships
# without the native `pet` (Python Environment Tools) finder, so environment discovery
# fails with `spawn .../python-env-tools/bin/pet ENOENT` and Jupyter hides kernels.
# This builds pet from Microsoft's MIT-licensed source (needs cargo) and copies it into
# every installed Python extension. Re-run after the extension updates.
set -euo pipefail

SRC="${PET_SRC:-$HOME/src/pet}"
EXT_ROOT="${VSCODIUM_EXTENSIONS:-$HOME/.vscode-oss/extensions}"

if [[ ! -x "$SRC/target/release/pet" ]]; then
  command -v cargo >/dev/null || { echo "cargo not found (brew install rust or rustup)"; exit 1; }
  [[ -d "$SRC/.git" ]] || git clone --depth 1 https://github.com/microsoft/python-environment-tools "$SRC"
  (cd "$SRC" && cargo build --release -p pet)
fi

shopt -s nullglob
installed=0
for ext in "$EXT_ROOT"/ms-python.python-*; do
  mkdir -p "$ext/python-env-tools/bin"
  cp "$SRC/target/release/pet" "$ext/python-env-tools/bin/pet"
  chmod +x "$ext/python-env-tools/bin/pet"
  echo "installed pet -> $ext"
  installed=1
done
[[ $installed == 1 ]] || { echo "no ms-python.python extension under $EXT_ROOT"; exit 1; }
echo "Now run 'Developer: Reload Window' in VSCodium."
