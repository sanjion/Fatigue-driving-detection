# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""utils/initialization."""

import contextlib
import platform
import threading


def emojis(str=""):
    """Returns an emoji-safe version of a string, stripped of emojis on Windows platforms."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    """A context manager and decorator for error handling that prints an optional message with emojis on exception."""

    def __init__(self, msg=""):
        """Initializes TryExcept with an optional message, used as a decorator or context manager for error handling."""
        self.msg = msg

    def __enter__(self):
        """Enter the runtime context related to this object for error handling with an optional message."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Context manager exit method that prints an error message with emojis if an exception occurred, always returns
        True.
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """Decorator @threaded to run a function in a separate thread, returning the thread instance."""

    def wrapper(*args, **kwargs):
        """Runs the decorated function in a separate daemon thread and returns the thread instance."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """
    Joins all daemon threads, optionally printing their names if verbose is True.

    Example: atexit.register(lambda: join_threads())
    """
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()



