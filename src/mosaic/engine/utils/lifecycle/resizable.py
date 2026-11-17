"""Resizable lifecycle module."""

from abc import ABC, abstractmethod


class Resizable(ABC):
    """Abstract class for classes that can be resized."""

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Method to call when resizing an object."""
