"""Window Module."""

from typing import TYPE_CHECKING

import glfw

from mosaic.engine.utils.lifecycle.closable import Closable

if TYPE_CHECKING:
    from ctypes import _Pointer


class Window(Closable):
    """Class for creating and managing a window."""

    _window: _Pointer[glfw._GLFWwindow]

    @property
    def window(self) -> _Pointer[glfw._GLFWwindow]:
        return self._window

    def __init__(self) -> None:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._window: _Pointer[glfw._GLFWwindow] = glfw.create_window(
            1280,
            720,
            "Mosaic",
            None,
            None,
        )

        if not self._window:
            glfw.terminate()
            error = "Failed to create GLFW window"
            raise RuntimeError(error)

    def close(self) -> None:
        """Close the window."""
        glfw.destroy_window(self._window)

    @property
    def should_close(self) -> bool:
        return glfw.window_should_close(self._window) != 0
