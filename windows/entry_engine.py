"""PyInstaller entry point for the engine (fpab.exe): the same CLI as the console script."""
import sys

from finalpass_audiobook.cli import main

sys.exit(main())
