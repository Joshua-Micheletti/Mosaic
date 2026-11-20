"""Window Abstract Class Module."""

from abc import abstractmethod

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.resizable import Resizable


class Window(Closable, Resizable):
    """Window Abstract class."""

    _title: str
    _width: int
    _height: int

    @property
    @abstractmethod
    def should_close(self) -> bool:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

    def __init__(self, title: str, width: int, height: int) -> None:
        self._title = title
        self._width = width
        self._height = height

    @abstractmethod
    def handle_resize(self) -> tuple[int, int] | None:
        """
        Call when checking if the window is resized. Returns its expected size and resets it.

        Returns:
            tuple[int, int] | None: Expected size or None in case no changes happened.

        """

    @abstractmethod
    def wait_for_framebuffer(self) -> bool:
        """
        Wait for the framebuffer to be available. This code blocks execution.

        Returns:
            bool: State of availability of the framebuffer

        """

    @abstractmethod
    def create_surface(self, instance_handle: int | None) -> int | None:
        """
        Create a surface starting from an instance handle.

        Args:
            instance_handle (int | None): Instance handle.

        Returns:
            int | None: Handle of the created surface.

        """
