"""Application module."""

from mosaic.engine.core.engine import Engine
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.runnable import Runnable


class Application(Closable, Runnable):
    """Class for starting the application."""

    def __init__(self) -> None:
        self.engine = Engine()

    def run(self) -> None:
        """Run the application."""
        try:
            self.engine.run()

        finally:
            self.close()

    def close(self) -> None:
        """Turn off the application."""
        # self.renderer.close()
        self.engine.close()
