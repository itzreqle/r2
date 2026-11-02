#!/usr/bin/env bash
# Build R2.app (release) with icon, Info.plist and ad-hoc signature.
#   macos/R2App/build_app.sh            → macos/R2App/build/R2.app
#   macos/R2App/build_app.sh --install  → also copy to ~/Applications
#   macos/R2App/build_app.sh --open     → build and launch
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
APP="$HERE/build/R2.app"
cd "$HERE"

# SwiftPM/Clang module caches embed the absolute checkout path. After the repo is
# moved or renamed they fail with "missing required module 'SwiftShims'", so start
# clean whenever .build was produced at a different path.
STAMP=".build/.r2-build-path"
if [[ -d .build && "$(cat "$STAMP" 2>/dev/null)" != "$HERE" ]]; then
  echo "▸ .build was created at another path; cleaning it"
  rm -rf .build
fi

echo "▸ swift build (release)"
LOG="$(mktemp -t r2-swift-build)"
if ! swift build -c release >"$LOG" 2>&1; then
  grep -E "error:|warning: .*error" "$LOG" || tail -20 "$LOG"
  echo "✗ swift build failed (full log: $LOG)"
  exit 1
fi
tail -1 "$LOG"; rm -f "$LOG"
echo "$HERE" > "$STAMP"

echo "▸ assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/R2App "$APP/Contents/MacOS/R2"

echo "▸ icon"
ICONSET="$HERE/build/AppIcon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
(cd "$REPO" && uv run python "$HERE/make_icon.py" "$HERE/build/icon_1024.png" >/dev/null)
for s in 16 32 128 256 512; do
  sips -z $s $s "$HERE/build/icon_1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s * 2))
  sips -z $d $d "$HERE/build/icon_1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>R2</string>
  <key>CFBundleDisplayName</key><string>R2</string>
  <key>CFBundleIdentifier</key><string>local.narrow-ai.r2</string>
  <key>CFBundleExecutable</key><string>R2</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.2.0</string>
  <key>CFBundleVersion</key><string>2</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSRequiresAquaSystemAppearance</key><false/>
  <key>NSMicrophoneUsageDescription</key>
  <string>R2 listens while you hold the talk button and transcribes your request on this Mac.</string>
  <key>R2RepoPath</key><string>$REPO</string>
</dict>
</plist>
PLIST

echo "▸ ad-hoc signing"
codesign --force --deep --sign - "$APP" 2>&1 | tail -1 || true

if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/R2.app"
  cp -R "$APP" "$HOME/Applications/R2.app"
  echo "▸ installed to ~/Applications/R2.app"
fi
echo "✓ built $APP"
if [[ "${1:-}" == "--open" ]]; then open "$APP"; fi
