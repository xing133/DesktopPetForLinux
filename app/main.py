from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.player_bridge import PlayerBridge
from app.runtime_paths import get_runtime_root
from app.ui.add_wife_wizard import AddWifeWizard
from app.ui.tray import DesktopDancerTray


def _show_wizard_front(wizard: AddWifeWizard) -> None:
    if wizard.isMinimized():
        wizard.showNormal()
    wizard.show()
    wizard.raise_()
    wizard.activateWindow()


def run() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    project_root = get_runtime_root()
    player_bridge = PlayerBridge(project_root=project_root)

    def on_dancer_ready(name: str) -> None:
        player_bridge.switch_to_dancer(name)

    wizard = AddWifeWizard(on_dancer_ready=on_dancer_ready)

    def on_add_wife() -> None:
        _show_wizard_front(wizard)

    def on_quit() -> None:
        player_bridge.stop()
        app.quit()

    tray = DesktopDancerTray(on_add_wife=on_add_wife, on_quit=on_quit)
    tray.show()
    print("[desktop-dancer] tray icon shown")
    app._desktop_dancer_tray = tray  # 防止托盘对象被回收

    # 启动即展示默认动画
    player_bridge.start_default_animation()

    app.aboutToQuit.connect(player_bridge.stop)
    return app.exec()


def run_add_wife_only() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    wizard = AddWifeWizard()
    _show_wizard_front(wizard)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
