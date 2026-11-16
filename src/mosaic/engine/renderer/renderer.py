from mosaic.engine.renderer.render_driver import RenderDriver
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.runnable import Runnable


class Renderer(Closable, Runnable):
    def __init__(self, driver: RenderDriver) -> None:
        self._driver = driver

    def run(self) -> None:
        self._driver.run()

    def close(self) -> None:
        self._driver.close()
