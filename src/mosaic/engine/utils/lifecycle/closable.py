"""Closable lifecycle module."""

from abc import ABC, abstractmethod


class Closable(ABC):
    """Abstract class for classes that can be closed."""

    @abstractmethod
    def close(self) -> None:
        """Method to call when closing an object."""
