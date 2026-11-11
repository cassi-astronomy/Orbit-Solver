"""One process-wide QApplication for the GUI tests, shut down before exit.

A QApplication kept only as a test-local variable may be garbage-collected
while widgets created later (or still referenced by Matplotlib canvases) are
alive; Linux/Qt then aborts during interpreter shutdown ("pure virtual method
called", SIGBUS/SIGSEGV) although every assertion passed, and the non-zero
exit code fails CI.  Every GUI test obtains the application here instead.
"""
import atexit
import os

_APPLICATION = None
_SHUTDOWN_REGISTERED = False


def qt_application():
    """Return the shared QApplication (offscreen unless a platform is set)."""
    global _APPLICATION, _SHUTDOWN_REGISTERED

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    _APPLICATION = QApplication.instance() or QApplication([])
    if not _SHUTDOWN_REGISTERED:
        atexit.register(_shutdown)
        _SHUTDOWN_REGISTERED = True
    return _APPLICATION


def dispose(*widgets) -> None:
    """Close and delete widgets now rather than at interpreter shutdown."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    for widget in widgets:
        if widget is not None:
            widget.close()
            widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application = QApplication.instance()
    if application is not None:
        application.processEvents()


def _shutdown() -> None:
    """Destroy Qt explicitly before PySide's interpreter-shutdown cleanup."""
    global _APPLICATION

    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None:
        return
    QApplication.closeAllWindows()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()
    application.shutdown()
    _APPLICATION = None
