"""Gather the licence texts of everything the Windows build bundles into one folder.

Called from fpab-windows.spec once PyInstaller has worked out what goes in, so the list is
exactly what ships: each bundled module is traced back to the installed distribution that
provides it, and that distribution's own licence files are copied. Python itself (and Tcl/Tk,
when the Tk frontend is in) comes from the interpreter's install. The installer puts the folder
at {app}\\Licenses, as the Mac app carries Contents/Resources/Licenses.
"""
from __future__ import annotations

import inspect
import shutil
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

NAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")
ALWAYS = ("finalpass-audiobook",  # this app (installed editable, so its own modules do not trace back to it)
          "pyinstaller")         # its bootloader is inside both exes (GPL with an exception that frees them)


def _top_level(dest: str) -> str:
    first = dest.replace("\\", "/").split("/")[0]
    return inspect.getmodulename(first) or first          # "_cffi_backend.cp312-win_amd64.pyd" -> "_cffi_backend"


def _take(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def _python_licence() -> Path:
    base = Path(sys.base_prefix)
    for p in (Path(sysconfig.get_paths()["stdlib"]) / "LICENSE.txt", base / "LICENSE.txt", base / "LICENSE"):
        if p.is_file():
            return p
    raise SystemExit(f"Python's own LICENSE.txt not found under {base}")


def _tcl_tk_licences() -> dict[str, Path]:
    """Tcl's and Tk's license.terms by library folder (tcl8.6, tk9.0, ...); some builds keep only the demos' copy."""
    found: dict[str, Path] = {}
    for p in sorted(Path(sys.base_prefix).rglob("license.terms"), key=lambda p: "demos" in p.parts):
        lib = next((q for q in p.relative_to(sys.base_prefix).parts if q.lower().startswith(("tcl", "tk"))), None)
        if lib:
            found.setdefault(lib, p)
    if not found:
        raise SystemExit("Tkinter is bundled but Tcl/Tk's license.terms was not found")
    return found


def gather(analyses, root: Path, out: Path, version: str) -> None:
    """Write the licence folder for the modules, binaries and data of these PyInstaller Analyses."""
    tops = {_top_level(name) for a in analyses for toc in (a.pure, a.binaries, a.datas) for name, *_ in toc}
    by_module = metadata.packages_distributions()
    dists = {d.lower() for t in tops for d in by_module.get(t, [])} | set(ALWAYS)

    shutil.rmtree(out, ignore_errors=True)
    rows = []
    for name in sorted(dists):
        dist = metadata.distribution(name)
        meta = dist.metadata
        files = [Path(dist.locate_file(f)) for f in (dist.files or [])
                 if ".dist-info" in str(f) and f.name.upper().startswith(NAMES)]
        label = f"{meta['Name']}-{meta['Version']}"
        for p in files:
            rel = p.relative_to(next(q for q in p.parents if q.name.endswith(".dist-info")))
            _take(p, out / label / rel)
        lic = meta.get("License-Expression") or ((meta.get("License") or "").splitlines() or [""])[0]
        rows.append(f"{meta['Name']} {meta['Version']}: {lic[:70] or 'see its folder'}"
                    + ("" if files else " (no licence file in its distribution)"))

    _take(_python_licence(), out / "Python-3.12" / "LICENSE.txt")
    rows.append("Python 3.12 (the interpreter and its standard library): PSF-2.0, see Python-3.12")
    if "tkinter" in tops:
        for lib, p in _tcl_tk_licences().items():
            _take(p, out / "Tcl-Tk" / lib / p.name)
        rows.append("Tcl/Tk (used by the window, through Python's tkinter): Tcl/Tk licence, see Tcl-Tk")
    if "soundfile" in dists:
        sf = metadata.distribution("soundfile")
        for f in sf.files or []:
            if f.parts[0] == "_soundfile_data" and f.name.upper().startswith(NAMES):
                _take(Path(sf.locate_file(f)), out / "libsndfile" / f.name)
        if not (out / "libsndfile").is_dir():
            raise SystemExit("soundfile is bundled but libsndfile's licence (_soundfile_data/COPYING) was not found")
        rows.append("libsndfile (bundled by soundfile, unmodified, dynamically loaded): LGPL-2.1-or-later; "
                    "source: https://github.com/libsndfile/libsndfile")
    vendor = root / "src" / "finalpass_audiobook" / "vendor"
    _take(vendor / "speech_truncation" / "PROVENANCE.md", out / "speech-truncation-detection-12M" / "PROVENANCE.md")
    _take(vendor / "speech_truncation" / "LICENSE", out / "speech-truncation-detection-12M" / "LICENSE")
    rows.append("mythicinfinity/speech-truncation-detection-12M (model weights and reference code): Apache-2.0")
    _take(vendor / "respiro" / "PROVENANCE.md", out / "Respiro-en" / "PROVENANCE.md")
    _take(vendor / "respiro" / "LICENSE", out / "Respiro-en" / "LICENSE")
    rows.append("Respiro-en (breath model reference code; the weights are fine-tuned from its published ones): MIT")

    (out / "README.txt").write_text(
        f"FinalPass AudioBook {version} for Windows contains the following software, each under its own licence.\n"
        "The full texts are in the folders beside this file.\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"licences gathered: {len(rows)} components")
    for r in rows:
        print("  " + r)
