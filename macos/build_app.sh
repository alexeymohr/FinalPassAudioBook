#!/usr/bin/env bash
# Build "FinalPass AudioBook.app": the SwiftUI window plus the engine it runs — a self-contained
# Python with fpab and its locked base dependencies (the chopped-word model runs as numpy; no
# PyTorch), and the verified model weights. The app is sandboxed with NO network entitlement.
#
#   macos/build_app.sh               -> macos/build/FinalPass AudioBook.app, ad-hoc signed (this Mac only)
#   macos/build_app.sh --sign "Developer ID Application: NAME (TEAM)"
#                                    -> signed for other Macs: Developer ID, secure timestamp, hardened
#                                       runtime; then macos/release.sh notarizes it and makes the DMG
#   macos/build_app.sh --test-hooks  -> plus the FPAB_AUTORUN hook for scripted checks (never released)
#
# Needs: this repo's .venv (for the interpreter), uv, and the model installed by `fpab setup-model`.
# Everything is assembled in a staging folder and moved into place only once it is signed and
# checked, so a failed build never leaves a half-built (or unsandboxed) app behind.
set -euo pipefail

TEST_HOOKS=0
SIGN_ID="-"                                 # ad hoc unless --sign names a Developer ID
while [ $# -gt 0 ]; do
    case "$1" in
        --test-hooks) TEST_HOOKS=1 ;;
        --sign) shift; SIGN_ID="${1:-}"; [ -n "$SIGN_ID" ] || { echo "--sign needs an identity" >&2; exit 2; } ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$SIGN_ID" = "-" ]; then
    SIGN=(-s - -f --timestamp=none)
else
    SIGN=(-s "$SIGN_ID" -f --timestamp --options runtime)
fi

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/.." && pwd -P)"
OUT="$HERE/build"
FINAL="$OUT/FinalPass AudioBook.app"
mkdir -p "$OUT"
STAGE="$(mktemp -d "$OUT/.stage.XXXXXX")"
APP="$STAGE/FinalPass AudioBook.app"
ENGINE="$APP/Contents/Resources/engine"
MODEL_SRC="${FPAB_MODEL_DIR:-$HOME/.cache/finalpass-audiobook}/speech-truncation-12M"
LOG="$STAGE/build.log"
trap 'rm -rf "$STAGE"' EXIT

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

EXCLUDE_NEWER="$(sed -n 's/^exclude-newer = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
VERSION="$(sed -n '/^\[project\]/,/^\[/s/^version = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
[ -f "$REPO/.venv/pyvenv.cfg" ] || die "no .venv in $REPO: run 'uv sync' first"
[ -d "$MODEL_SRC" ] || die "model not installed ($MODEL_SRC): run 'fpab setup-model' first"
[ -n "$EXCLUDE_NEWER" ] || die "no [tool.uv] exclude-newer in pyproject.toml"
[ -n "$VERSION" ] || die "no [project] version in pyproject.toml"
if [ "$SIGN_ID" != "-" ]; then
    [[ "$SIGN_ID" == "Developer ID Application: "* ]] || die "--sign needs a \"Developer ID Application: …\" identity"
    security find-identity -v -p codesigning | grep -qF "\"$SIGN_ID\"" || die "no valid signing identity \"$SIGN_ID\""
fi

# The 7-day package hold: the lock's cutoff must be at least 7 days in the past.
"$REPO/.venv/bin/python" -I -c "
import sys, datetime as d
cut = d.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
age = d.datetime.now(d.timezone.utc) - cut
sys.exit(0 if age >= d.timedelta(days=7) else f'exclude-newer {sys.argv[1]} is only {age.days} days ago (7-day hold)')
" "$EXCLUDE_NEWER"

if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "note: the working tree has uncommitted changes; they are what gets bundled"
fi
# Every file in the package that is not ignored goes into the app: an untracked one (a note, a dump
# from calibration) must never ride along unseen.
untracked="$(git -C "$REPO" status --porcelain --untracked-files=all -- src pyproject.toml 2>/dev/null | grep '^??' || true)"
[ -z "$untracked" ] || { echo "$untracked"; die "untracked files under src/ would be bundled: commit, remove or ignore them"; }

# The real interpreter directory behind the venv. uv's version folders are symlinks and everything
# copied here gets re-signed, so resolve fully (never sign through a link) and refuse anything odd:
# a missing interpreter must never turn into "copy /".
HOME_LINE="$(sed -n 's/^home = //p' "$REPO/.venv/pyvenv.cfg" | head -1)"
[ -n "$HOME_LINE" ] || die ".venv/pyvenv.cfg has no 'home =' line"
PY_BIN_DIR="$(readlink -f "$HOME_LINE" 2>/dev/null || true)"
[ -n "$PY_BIN_DIR" ] && [ -x "$PY_BIN_DIR/python3.12" ] || die "the venv's interpreter ($HOME_LINE) is missing"
PY_ROOT="$(cd "$PY_BIN_DIR/.." && pwd -P)"
case "$PY_ROOT" in
    "$HOME"/.local/share/uv/python/cpython-3.12*) ;;
    *) die "unexpected interpreter location: $PY_ROOT" ;;
esac

say "Swift app"
SWIFT_FLAGS=()
[ "$TEST_HOOKS" = 1 ] && SWIFT_FLAGS=(-Xswiftc -DFPAB_TEST_HOOKS)
swift build -c release --package-path "$HERE" ${SWIFT_FLAGS[@]+"${SWIFT_FLAGS[@]}"}
BIN="$(swift build -c release --package-path "$HERE" ${SWIFT_FLAGS[@]+"${SWIFT_FLAGS[@]}"} --show-bin-path)/FPAB"
mkdir -p "$APP/Contents/MacOS" "$ENGINE/model"
cp "$BIN" "$APP/Contents/MacOS/FPAB"
strip -S -x "$APP/Contents/MacOS/FPAB"      # debug records name the source folder; signed with the app below
cp "$HERE/Resources/Info.plist" "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$APP/Contents/Info.plist"

say "App icon (Icon Composer file -> Assets.car, plus an .icns for macOS before 26)"
xcrun actool --compile "$APP/Contents/Resources" --platform macosx --minimum-deployment-target 14.0 \
    --app-icon AppIcon --output-partial-info-plist "$STAGE/icon.plist" --errors --warnings \
    "$HERE/Resources/AppIcon.icon" > "$STAGE/actool.log" || { cat "$STAGE/actool.log"; die "icon compile failed"; }
[ -s "$APP/Contents/Resources/Assets.car" ] && [ -s "$APP/Contents/Resources/AppIcon.icns" ] \
    || { cat "$STAGE/actool.log"; die "icon compile produced no icon"; }

say "Python interpreter (copied from $PY_ROOT)"
ditto "$PY_ROOT" "$ENGINE/python"
PY="$ENGINE/python/bin/python3.12"
# Nothing in the app may name this Mac's folders (checked before signing). The library's own name
# pointed at the uv folder it was copied from: name it by its place in the app, and re-sign it at
# once (arm64 code with a broken signature is not loaded).
LIBPY="$ENGINE/python/lib/libpython3.12.dylib"
[[ ! -L "$LIBPY" && "$(readlink -f "$LIBPY")" == "$ENGINE"/* ]] || die "unexpected libpython: $LIBPY"
install_name_tool -id @rpath/libpython3.12.dylib "$LIBPY" 2>/dev/null || die "install_name_tool failed"
codesign -s - -f "$LIBPY" 2>/dev/null || die "re-sign after install_name_tool failed"
SHOWN="/Applications/FinalPass AudioBook.app/Contents/Resources/engine"

say "Locked base dependencies (uv.lock hashes, published before $EXCLUDE_NEWER)"
FP_REQ="$(uv export --project "$REPO" --frozen --no-dev --no-hashes --no-emit-project --no-header 2>/dev/null \
          | grep '^finalpass @ git+' | head -1)"
[ -n "$FP_REQ" ] || die "finalpass git pin not found in the lock"
uv export --project "$REPO" --frozen --no-dev --no-emit-project --no-emit-package finalpass --no-header \
    -o "$STAGE/requirements.txt" >/dev/null
uv pip install --python "$PY" --break-system-packages --no-deps --require-hashes \
    --exclude-newer "$EXCLUDE_NEWER" -r "$STAGE/requirements.txt"
# The two built from source get their build backend from hash-pinned build constraints.
uv pip install --python "$PY" --break-system-packages --no-deps --exclude-newer "$EXCLUDE_NEWER" \
    --build-constraints "$HERE/build-constraints.txt" "$FP_REQ"
uv pip install --python "$PY" --break-system-packages --no-deps --exclude-newer "$EXCLUDE_NEWER" \
    --build-constraints "$HERE/build-constraints.txt" "$REPO"
# pip's record of where fpab was installed from names this Mac's folders; the app does not need it
for d in "$ENGINE"/python/lib/python3.12/site-packages/finalpass_audiobook-*.dist-info; do
    rm -f "$d/direct_url.json"
    /usr/bin/sed -i '' '/direct_url.json/d' "$d/RECORD"     # BSD sed (a GNU sed first in PATH reads -i differently)
done

say "Model weights (the one verified file, nothing else from the cache)"
WEIGHTS_REL="$("$REPO/.venv/bin/python" -I -c "from finalpass_audiobook import model; print(model.REVISION[:12])")/model.safetensors"
[ -f "$MODEL_SRC/$WEIGHTS_REL" ] || die "model not installed ($MODEL_SRC/$WEIGHTS_REL): run 'fpab setup-model' first"
mkdir -p "$(dirname "$ENGINE/model/speech-truncation-12M/$WEIGHTS_REL")"
cp "$MODEL_SRC/$WEIGHTS_REL" "$ENGINE/model/speech-truncation-12M/$WEIGHTS_REL"
FPAB_MODEL_DIR="$ENGINE/model" "$PY" -I -c "from finalpass_audiobook import model; model.load(); print('weights verified')"

say "Trim what an analysis run never uses"
LIB="$ENGINE/python/lib"
STD="$LIB/python3.12"
SITE="$STD/site-packages"
rm -rf "$STD"/{test,idlelib,tkinter,turtledemo,ensurepip,lib2to3} "$STD"/turtle.py \
       "$STD"/lib-dynload/_tkinter*.so "$LIB"/tcl* "$LIB"/tk* "$LIB"/itcl* "$LIB"/libtcl* "$LIB"/libtk* \
       "$STD"/config-3.12-darwin "$ENGINE/python/include" "$ENGINE/python/share" "$LIB"/pkgconfig \
       "$SITE"/pip "$SITE"/pip-*.dist-info "$SITE"/setuptools "$SITE"/setuptools-*.dist-info \
       "$SITE"/_distutils_hack "$SITE"/pkg_resources "$SITE"/distutils-precedence.pth \
       "$SITE"/pygments "$SITE"/pygments-*.dist-info   # only rich's traceback/syntax use it; fpab uses neither
find "$ENGINE/python/bin" -type f ! -name 'python3.12' -delete
find "$ENGINE/python/bin" -type l ! -name 'python3' ! -name 'python3.12' -delete
find "$SITE" -depth -type d \( -name tests -o -name testing -o -name __pycache__ \) -path '*/scipy/*' -exec rm -rf {} +
find "$SITE"/numpy -depth -type d -name tests -exec rm -rf {} +
find "$ENGINE" -name '*.pyi' -delete
# Standard-library parts no run imports (the imports a run makes were recorded; the few lazy uses by
# dependencies — scipy's multiprocess map, click's browser launch — are functions fpab never calls).
rm -rf "$STD"/{pydoc_data,asyncio,xml,xmlrpc,multiprocessing,sqlite3,curses,dbm,wsgiref,venv,__phello__} \
       "$STD"/{_pydecimal,_pydatetime,_pyio,doctest,pdb,bdb,tarfile,pickletools,webbrowser,imaplib,nntplib}.py \
       "$STD"/{poplib,smtplib,ftplib,telnetlib,mailbox,mailcap,cgi,cgitb,xdrlib,uu,sndhdr,imghdr,aifc}.py \
       "$STD"/{sunau,chunk,wave,pipes,crypt,profile,cProfile,pstats,trace,tabnanny,pyclbr,modulefinder}.py \
       "$STD"/{zipapp,antigravity,this,__hello__,sched,shelve,rlcompleter}.py \
       "$STD"/lib-dynload/{_dbm,_crypt}.cpython-312-darwin.so

say "Strip local symbols from every compiled file (they only serve debuggers)"
# strip invalidates a binary's signature and macOS kills unsigned code, so each stripped file gets a
# plain ad-hoc signature at once (the real signing, with entitlements, comes later). Only regular
# files inside the bundle, never through a link.
before=$(du -sk "$ENGINE" | cut -f1)
while IFS= read -r f; do
    [[ -L "$f" ]] && continue
    real="$(readlink -f "$f")"
    [[ "$real" == "$ENGINE"/* ]] || die "refusing to strip outside the bundle: $f -> $real"
    strip -x "$real" 2>/dev/null || die "strip failed: $real"
    codesign -s - -f "$real" 2>/dev/null || die "re-sign after strip failed: $real"
done < <(find "$ENGINE" -type f \( -name '*.so' -o -name '*.dylib' -o -path '*/bin/python3.12' \))
echo "stripped: $(( (before - $(du -sk "$ENGINE" | cut -f1)) / 1024 )) MB"
FPAB_MODEL_DIR="$ENGINE/model" "$PY" -I -c "import finalpass_audiobook.cli, finalpass_audiobook.run, scipy.signal, scipy.ndimage, soundfile, click._termui_impl, rich.progress, pydoc, unittest; print('imports ok')"

say "Smoke checks on error paths (before signing: once signed, the engine only runs inside the app)"
SMOKE="$STAGE/smoke"
mkdir -p "$SMOKE"
FPAB=(-I -c "import sys; sys.argv[0] = 'fpab'; from finalpass_audiobook.cli import main; main()")
"$PY" "${FPAB[@]}" --help >/dev/null
printf 'not audio' > "$SMOKE/broken.wav"
FPAB_MODEL_DIR="$ENGINE/model" "$PY" "${FPAB[@]}" check --csv-dir "$SMOKE/o" --progress jsonl -- "$SMOKE/broken.wav" \
    > "$SMOKE/broken.jsonl" 2>"$SMOKE/broken.err" || { cat "$SMOKE/broken.err"; die "a broken WAV crashed the engine"; }
grep -q '"csv": null' "$SMOKE/broken.jsonl" || die "a broken WAV was not reported as skipped"
"$PY" -I -c "import sys, numpy as np, soundfile as sf
sf.write(sys.argv[1], 0.01 * np.random.default_rng(1).standard_normal(44100 * 3), 44100, subtype='PCM_24')" "$SMOKE/ok.wav"
FPAB_MODEL_DIR="$SMOKE/no-model" "$PY" "${FPAB[@]}" check --out "$SMOKE/rep" -- "$SMOKE/ok.wav" >/dev/null 2>&1
grep -q "truncation check skipped" "$SMOKE/rep/issues.txt" || die "a missing model was not reported"
find "$ENGINE" -name '__pycache__' -type d -prune -exec rm -rf {} +
echo "error paths ok"

say "Precompile what a run imports (one run on a synthetic WAV; the sandboxed engine never writes .pyc)"
# The interpreter's build-time settings name the uv folder it came from (no run reads them).
for f in "$ENGINE"/python/lib/python3.12/_sysconfigdata_*.py; do
    PY_ROOT="$PY_ROOT" SHOWN="$SHOWN/python" "$REPO/.venv/bin/python" -I -c "
import os, sys; p = sys.argv[1]; t = open(p, encoding='utf-8').read()
open(p, 'w', encoding='utf-8').write(t.replace(os.environ['PY_ROOT'], os.environ['SHOWN']))" "$f"
done
WARM="$STAGE/warm"
mkdir -p "$WARM"
"$PY" -I -c "import sys, numpy as np, soundfile as sf
sr = 44100; t = np.arange(sr * 12) / sr
x = 0.05 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0) + 1e-4 * np.random.default_rng(0).standard_normal(len(t))
sf.write(sys.argv[1], x, sr, subtype='PCM_24')" "$WARM/warm.wav"
FPAB_MODEL_DIR="$ENGINE/model" "$PY" "${FPAB[@]}" check --csv-dir "$WARM/out" --with-pauses --progress jsonl \
    -- "$WARM/warm.wav" >/dev/null
[ -f "$WARM/out/warm.csv" ] || die "warm-up run wrote no CSV"
# Compiled files record where their source was: recompile each under the path it has in the app.
"$PY" -I - "$ENGINE" "$SHOWN" <<'PYC'
import os, py_compile, sys
engine, shown = sys.argv[1], sys.argv[2]
for root, _, files in os.walk(engine):
    if os.path.basename(root) != "__pycache__":
        continue
    for f in files:
        src = os.path.join(os.path.dirname(root), f.split(".cpython-")[0] + ".py")
        if f.endswith(".pyc") and os.path.isfile(src):
            py_compile.compile(src, cfile=os.path.join(root, f), dfile=os.path.join(shown, os.path.relpath(src, engine)),
                               doraise=True)
PYC
echo "compiled modules: $(find "$ENGINE" -name '*.pyc' | wc -l | tr -d ' ')"
# The folders this build read from, as fixed strings (third-party files name their own build machines).
leaked="$(grep -rlF -e "$REPO" -e "$PY_ROOT" -e "$MODEL_SRC" -e "$HOME/.cache" -e "$HOME/.local" "$APP" 2>/dev/null \
          | head -5 || true)"                                   # grep finds none: exit 1
[ -z "$leaked" ] || { echo "$leaked"; die "the app would carry this Mac's folder names"; }

say "Licences: each bundled component's own licence texts, gathered in Contents/Resources/Licenses"
"$REPO/.venv/bin/python" -I - "$SITE" "$STD" "$APP/Contents/Resources/Licenses" "$VERSION" <<'PYL'
import shutil, sys
from pathlib import Path
site, std, out, version = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
NAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")
rows = []
def take(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
for d in sorted(site.glob("*.dist-info")):
    meta = {}
    for line in (d / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            break
        k, _, v = line.partition(": ")
        meta.setdefault(k, v.strip())
    name, ver = meta.get("Name", d.name), meta.get("Version", "")
    files = [p for p in sorted(d.rglob("*")) if p.is_file() and p.name.upper().startswith(NAMES)]
    for p in files:
        take(p, out / f"{name}-{ver}" / p.relative_to(d))
    lic = meta.get("License-Expression") or (meta.get("License", "").splitlines() or [""])[0]
    rows.append(f"{name} {ver}: {lic[:70] or 'see its folder'}")
take(std / "LICENSE.txt", out / "Python-3.12" / "LICENSE.txt")
rows.append("Python 3.12 (the interpreter and its standard library): PSF-2.0, see Python-3.12")
take(site / "_soundfile_data" / "COPYING", out / "libsndfile" / "COPYING")
rows.append("libsndfile (bundled by soundfile, unmodified, dynamically loaded): LGPL-2.1-or-later; "
            "source: https://github.com/libsndfile/libsndfile")
vendor = site / "finalpass_audiobook" / "vendor" / "speech_truncation"
take(vendor / "LICENSE", out / "speech-truncation-detection-12M" / "LICENSE")
take(vendor / "PROVENANCE.md", out / "speech-truncation-detection-12M" / "PROVENANCE.md")
rows.append("mythicinfinity/speech-truncation-detection-12M (model weights and reference code): Apache-2.0")
(out / "README.txt").write_text(
    f"FinalPass AudioBook {version} contains the following software, each under its own licence.\n"
    "The full texts are in the folders beside this file.\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
print(f"licences gathered: {len(rows)} components")
PYL

say "Guard: no link may lead outside the bundle or nowhere"
leaks="$(find "$APP" -type l | while IFS= read -r l; do
    t="$(readlink -f "$l" || true)"
    [[ "$t" == "$APP"/* && -e "$l" ]] || echo "$l -> $t"
done)"
[ -z "$leaks" ] || { echo "$leaks"; die "links leave the bundle"; }

say "Sign ($([ "$SIGN_ID" = "-" ] && echo "ad hoc" || echo "$SIGN_ID, hardened runtime")): every Mach-O file, found by its magic number"
"$REPO/.venv/bin/python" -I - "$ENGINE" > "$STAGE/macho.txt" <<'PY'
import os, sys
MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}
for root, dirs, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p):
            continue
        with open(p, "rb") as fh:
            if fh.read(4) in MAGIC:
                print(p)
PY
count=0
while IFS= read -r f; do
    [ "$f" = "$PY" ] && continue
    codesign "${SIGN[@]}" "$f" >>"$LOG" 2>&1 || { tail -5 "$LOG"; die "signing failed: $f"; }
    count=$((count + 1))
done < "$STAGE/macho.txt"
codesign "${SIGN[@]}" --entitlements "$HERE/Resources/Engine.entitlements" "$PY" >>"$LOG" 2>&1 \
    || { tail -5 "$LOG"; die "signing the engine failed"; }
codesign "${SIGN[@]}" --entitlements "$HERE/Resources/FPAB.entitlements" "$APP" >>"$LOG" 2>&1 \
    || { tail -5 "$LOG"; die "signing the app failed"; }
echo "signed $count libraries, the engine and the app"

say "Verify"
codesign --verify --strict "$APP" || die "the app's signature does not verify"
while IFS= read -r f; do
    codesign --verify --strict "$f" >>"$LOG" 2>&1 || { tail -3 "$LOG"; die "does not verify: $f"; }
done < "$STAGE/macho.txt"
keys() {
    codesign -d --entitlements - --xml "$1" 2>/dev/null | plutil -convert json -o - - 2>/dev/null \
        | "$REPO/.venv/bin/python" -I -c "import json, sys; print(','.join(sorted(k for k, v in json.load(sys.stdin).items() if v)))"
}
[ "$(keys "$APP")" = "com.apple.security.app-sandbox,com.apple.security.files.bookmarks.app-scope,com.apple.security.files.user-selected.read-write" ] \
    || die "the app's entitlements are not exactly sandbox + user-selected files + bookmarks: $(keys "$APP")"
[ "$(keys "$PY")" = "com.apple.security.app-sandbox,com.apple.security.inherit" ] \
    || die "the engine's entitlements are not exactly sandbox + inherit: $(keys "$PY")"
echo "entitlements exact; no network"
if [ "$SIGN_ID" != "-" ]; then                  # what notarization requires of every executable
    for f in "$APP" "$PY"; do
        info="$(codesign -dvv "$f" 2>&1)"
        grep -q "^Authority=$SIGN_ID\$" <<<"$info" || die "not signed by $SIGN_ID: $f"
        grep -q "(runtime)" <<<"$info" || die "no hardened runtime: $f"
        grep -q "^Timestamp=" <<<"$info" || die "no secure timestamp: $f"
    done
    echo "Developer ID, hardened runtime and secure timestamp on the app and the engine"
fi

rm -rf "$FINAL"
mv "$APP" "$FINAL"
du -sh "$FINAL"
echo "Built: $FINAL$([ "$TEST_HOOKS" = 1 ] && echo '  (with test hooks)')$([ "$SIGN_ID" != "-" ] && echo "
Signed by $SIGN_ID. Next: macos/release.sh (notarize, staple, DMG)")"
