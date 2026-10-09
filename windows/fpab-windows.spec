# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: build fpab-gui.exe (Tkinter) and fpab.exe (engine) into one onedir.

The engine's models are NOT frozen in here; the installer drops them beside the exes in
`models/`, where the GUI finds them and points the engine at them via FPAB_MODEL_DIR.
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent            # repo root; this spec lives in windows/
GUI = str(ROOT / "windows" / "entry_gui.py")
ENGINE = str(ROOT / "windows" / "entry_engine.py")

datas, binaries, hiddenimports = [], [], []
for package in ("finalpass_audiobook", "finalpass"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden
datas += collect_data_files("soundfile")
binaries += collect_dynamic_libs("soundfile")
hiddenimports += ["soundfile", "scipy", "numpy"]

try:                                              # optional: Explorer drag-and-drop
    datas += collect_data_files("tkinterdnd2")
    hiddenimports.append("tkinterdnd2")
except Exception:
    pass

EXCLUDES = ["torch", "torchaudio", "transformers", "safetensors", "pytest", "PIL", "matplotlib",
            "gi", "gi.repository"]                # GTK is never imported on Windows (the Tk frontend is used)


def make_analysis(script):
    return Analysis([script], pathex=[str(ROOT), str(ROOT / "gui")],
                    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
                    excludes=EXCLUDES, noarchive=False)


gui = make_analysis(GUI)
engine = make_analysis(ENGINE)

sys.path.insert(0, str(ROOT / "windows"))
import finalpass_audiobook
import licenses                                   # windows/licenses.py: the installer's {app}\Licenses

licenses.gather([gui, engine], ROOT, ROOT / "windows" / "staging" / "Licenses", finalpass_audiobook.__version__)

exe_gui = EXE(PYZ(gui.pure), gui.scripts, [], [], name="fpab-gui", console=False,
              exclude_binaries=True, upx=False, strip=False)
exe_engine = EXE(PYZ(engine.pure), engine.scripts, [], [], name="fpab", console=True,
                 exclude_binaries=True, upx=False, strip=False)

COLLECT(exe_gui, gui.binaries, gui.datas,
        exe_engine, engine.binaries, engine.datas,
        name="FinalPassAudioBook", upx=False, strip=False)
