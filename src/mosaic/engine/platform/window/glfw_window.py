"""GLFW Window Module."""

from ctypes import byref, c_void_p
from typing import TYPE_CHECKING, override

import glfw

from mosaic.engine.platform.window.window import Window

if TYPE_CHECKING:
    from ctypes import _Pointer


class GLFWWindow(Window):
    """Class for creating and managing a window."""

    _window: _Pointer[glfw._GLFWwindow]
    _pending_resize: tuple[int, int] | None

    @property
    def window(self) -> _Pointer[glfw._GLFWwindow]:
        return self._window

    @property
    @override
    def should_close(self) -> bool:
        return glfw.window_should_close(self._window) != 0

    @property
    @override
    def width(self) -> int:
        return glfw.get_framebuffer_size(self._window)[0]

    @property
    @override
    def height(self) -> int:
        return glfw.get_framebuffer_size(self._window)[1]

    @property
    @override
    def size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self._window)

    def __init__(self, title: str, width: int, height: int) -> None:
        super().__init__(title, width, height)

        self._pending_resize = None
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._window: _Pointer[glfw._GLFWwindow] = glfw.create_window(
            self._width,
            self._height,
            self._title,
            None,
            None,
        )

        if not self._window:
            glfw.terminate()
            error = "Failed to create GLFW window"
            raise RuntimeError(error)

        glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_resize)

    @override
    def handle_resize(self) -> tuple[int, int] | None:
        current_value = self._pending_resize

        self._resize = None

        return current_value

    @override
    def wait_for_framebuffer(self) -> bool:
        """Wait until the window can provide a non-zero swapchain extent."""
        width, height = glfw.get_framebuffer_size(self._window)
        while width == 0 or height == 0:
            if glfw.window_should_close(self._window):
                return False
            glfw.wait_events()
            width, height = glfw.get_framebuffer_size(self._window)
        return True

    @override
    def close(self) -> None:
        """Close the window."""
        glfw.destroy_window(self._window)

    @override
    def resize(self, width: int, height: int) -> None:
        pass

    @override
    def create_surface(self, instance_handle: int | None) -> int | None:
        surface_handle = c_void_p(0)
        result = glfw.create_window_surface(
            instance_handle, self._window, None, byref(surface_handle)
        )
        if result != 0:
            error = f"glfwCreateWindowSurface failed: {result}"
            raise RuntimeError(error)

        return surface_handle.value

    def _on_framebuffer_resize(
        self, _window: _Pointer[glfw._GLFWwindow], width: int, height: int
    ) -> None:
        self._pending_resize = (width, height)
