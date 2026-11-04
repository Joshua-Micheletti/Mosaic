"""Platform module."""

import glfw

from mosaic.engine.platform.window import Window


class Platform:
    """Class for managing platform specific implementations like window management and inputs."""

    def __init__(self) -> None:
        if not glfw.init():
            error = "Failed to initialize GLFW"
            raise RuntimeError(error)

        self._window = Window()

    def poll_events(self) -> None:
        """Poll the events from the platform."""
        glfw.poll_events()

    def close(self) -> None:
        """Close the platform."""
        self._window.close()
        glfw.terminate()

    @property
    def window(self) -> Window:
        return self._window
