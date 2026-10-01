#!/usr/bin/env bash
# Turn the Developer-ID-signed app into a download: notarize it with Apple, staple the ticket into
# it (so it opens even offline), and wrap it in a DMG that is itself signed, notarized and stapled.
#
#   macos/build_app.sh --sign "Developer ID Application: NAME (TEAM)"
#   macos/release.sh [--profile NAME]   -> macos/build/FinalPass-AudioBook-<version>.dmg
#
# Needs a notarization profile in the keychain, made once by the developer (it asks for an
# app-specific password; this script never sees it):
#   xcrun notarytool store-credentials fpab-notary --apple-id APPLE_ID --team-id TEAM
# Notarization uploads the app to Apple's notary service; the build has checked that it carries
# no file naming the building Mac's folders.
set -euo pipefail

PROFILE="${FPAB_NOTARY_PROFILE:-fpab-notary}"
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) shift; PROFILE="${1:?--profile needs a name}" ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/.." && pwd -P)"
APP="$HERE/build/FinalPass AudioBook.app"
say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -d "$APP" ] || die "no app at $APP: build it with macos/build_app.sh --sign …"
VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")"
INFO="$(codesign -dvv "$APP" 2>&1)"
ID="$(sed -n 's/^Authority=\(Developer ID Application: .*\)$/\1/p' <<<"$INFO" | head -1)"
[ -n "$ID" ] || die "the app is not signed with a Developer ID: build it with macos/build_app.sh --sign …"
grep -q "(runtime)" <<<"$INFO" || die "the app has no hardened runtime"
grep -rqF FPAB_AUTORUN "$APP/Contents/MacOS" && die "this is a test-hook build: never released"
codesign --verify --deep --strict "$APP" || die "the app's signature does not verify"
xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1 \
    || die "no notarization profile \"$PROFILE\" in the keychain (see the top of this script)"

WORK="$(mktemp -d "$HERE/build/.release.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

notarize() {                                    # $1: a zip or a DMG
    "$REPO/.venv/bin/python" -I -c "import sys" >/dev/null || die "the repo's .venv is needed"
    xcrun notarytool submit "$1" --keychain-profile "$PROFILE" --wait --output-format json > "$WORK/submit.json" \
        || true
    local status id
    status="$("$REPO/.venv/bin/python" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "$WORK/submit.json" 2>/dev/null || true)"
    id="$("$REPO/.venv/bin/python" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('id',''))" "$WORK/submit.json" 2>/dev/null || true)"
    if [ "$status" != "Accepted" ]; then
        cat "$WORK/submit.json" >&2 || true
        [ -n "$id" ] && xcrun notarytool log "$id" --keychain-profile "$PROFILE" >&2 || true
        die "notarization of $(basename "$1") was not accepted (status: ${status:-none})"
    fi
    echo "notarized: $(basename "$1") ($id)"
}

say "Notarize the app ($ID, version $VERSION)"
ditto -c -k --keepParent "$APP" "$WORK/app.zip"
notarize "$WORK/app.zip"
xcrun stapler staple "$APP" >/dev/null && xcrun stapler validate "$APP" >/dev/null || die "stapling the app failed"
echo "ticket stapled to the app"

say "DMG: the stapled app and a link to Applications"
mkdir "$WORK/dmg"
ditto "$APP" "$WORK/dmg/FinalPass AudioBook.app"
ln -s /Applications "$WORK/dmg/Applications"
DMG="$HERE/build/FinalPass-AudioBook-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "FinalPass AudioBook $VERSION" -srcfolder "$WORK/dmg" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null
codesign -s "$ID" --timestamp "$DMG" || die "signing the DMG failed"
notarize "$DMG"
xcrun stapler staple "$DMG" >/dev/null && xcrun stapler validate "$DMG" >/dev/null || die "stapling the DMG failed"

say "Check as a downloader's Mac would (Gatekeeper)"
spctl -a -vv -t exec "$APP" 2>&1 | sed 's/^/  /'
spctl -a -vv -t open --context context:primary-signature "$DMG" 2>&1 | sed 's/^/  /'
spctl -a -t exec "$APP" && spctl -a -t open --context context:primary-signature "$DMG" \
    || die "Gatekeeper does not accept the release"
du -h "$DMG" | sed 's/^/  /'
echo "  sha256 $(shasum -a 256 "$DMG" | cut -d' ' -f1)"
echo "Release: $DMG"
