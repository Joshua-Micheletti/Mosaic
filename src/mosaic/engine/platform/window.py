"""Window Module."""

import glfw

from mosaic.engine.utils.lifecycle.closable import Closable


class Window(Closable):
    """Class for creating and managing a window."""

    def __init__(self) -> None:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._window = glfw.create_window(
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
