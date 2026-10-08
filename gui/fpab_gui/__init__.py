"""FinalPass AudioBook — desktop GUI (GTK4 frontend over the `fpab` engine).

The core modules (`engine`, `placement`, `config`, `runmodel`) are toolkit-agnostic
pure Python so another frontend (Qt, tk, web) can reuse them; `gtk_app` is the GTK4 one.
"""

__version__ = "0.1.0"
