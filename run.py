from __future__ import annotations


if __name__ == "__main__":
    import ctypes
    import faulthandler
    import sys
    import traceback
    from pathlib import Path

    log_file = Path(__file__).with_name("orbit-crash.log")
    native_log = log_file.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(native_log, all_threads=True)

    def report_exception(exc_type, exc_value, exc_traceback) -> None:
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(details)
        message = (
            f"Orbit Solver narazil na chybu:\n\n{exc_value}\n\n"
            f"Podrobnosti: {log_file}"
        )
        if sys.platform == "win32":
            ctypes.windll.user32.MessageBoxW(
                0, message, "Orbit Solver — chyba", 0x10,
            )
        else:
            print(message, file=sys.stderr)

    sys.excepthook = report_exception
    try:
        from orbit_app.main_window import run

        raise SystemExit(run())
    except SystemExit:
        raise
    except Exception:
        report_exception(*sys.exc_info())
        raise SystemExit(1)

