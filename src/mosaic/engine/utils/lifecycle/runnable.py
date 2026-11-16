"""Runnable lifecycle module."""

from abc import ABC, abstractmethod


class Runnable(ABC):
    """Abstract class for classes that can be executed."""

    @abstractmethod
    def run(self) -> None:
        """Method to call when running an object."""
