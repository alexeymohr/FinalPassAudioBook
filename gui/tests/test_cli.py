from fpab_gui import cli


def test_default_ui_is_tk_on_windows(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.delenv("FPAB_GUI_UI", raising=False)
    assert cli._default_ui() == "tk"


def test_default_ui_honours_env(monkeypatch):
    monkeypatch.setenv("FPAB_GUI_UI", "tk")
    assert cli._default_ui() == "tk"
    monkeypatch.setenv("FPAB_GUI_UI", "gtk")
    assert cli._default_ui() == "gtk"
