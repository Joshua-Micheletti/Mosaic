"""Renderer Module."""

from typing import TYPE_CHECKING, override

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.resizable import Resizable
from mosaic.engine.utils.lifecycle.runnable import Runnable

if TYPE_CHECKING:
    from mosaic.engine.renderer.render_driver import RenderDriver


class Renderer(Closable, Runnable, Resizable):
    """Renderer Class."""

    def __init__(self, driver: RenderDriver) -> None:
        self._driver = driver

    @override
    def run(self) -> None:
        self._driver.run()

    @override
    def close(self) -> None:
        self._driver.close()

    @override
    def resize(self, width: int = 0, height: int = 0) -> None:
        self._driver.resize(width, height)
