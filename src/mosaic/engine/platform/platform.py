"""Platform module."""

import glfw

from mosaic.engine.platform.backend.vulkan.vulkan_backend import VulkanBackend
from mosaic.engine.platform.window import Window


class Platform:
    """Class for managing platform specific implementations like window management and inputs."""

    @property
    def window(self) -> Window:
        return self._window

    @property
    def backend(self) -> VulkanBackend:
        return self._backend

    def __init__(self) -> None:
        if not glfw.init():
            error = "Failed to initialize GLFW"
            raise RuntimeError(error)

        self._window = Window()
        self._backend = VulkanBackend(self._window.window)

    def poll_events(self) -> None:
        """Poll the events from the platform."""
        glfw.poll_events()

    def close(self) -> None:
        """Close the platform."""
        if hasattr(self, "_backend"):
            self._backend.close()
        if hasattr(self, "_window"):
            self._window.close()
        glfw.terminate()
