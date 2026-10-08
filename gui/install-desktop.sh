#!/usr/bin/env bash
# Add a desktop launcher for the FinalPass AudioBook GUI.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$apps"
cat > "$apps/fpab-gui.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=FinalPass AudioBook
Comment=Local, offline QC for AI-narrated audiobook chapters
Exec="$here/fpab-gui"
Terminal=false
Categories=AudioVideo;Audio;
Keywords=audiobook;QC;audio;
EOF
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$apps"
fi
echo "Installed $apps/fpab-gui.desktop"
