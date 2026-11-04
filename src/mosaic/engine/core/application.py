"""Application module."""

from mosaic.engine.platform.platform import Platform
from mosaic.engine.utils.lifecycle.closable import Closable


class Application(Closable):
    """Class for starting the application."""

    def __init__(self) -> None:
        self.platform = Platform()

    def run(self) -> None:
        """Run the application."""
        try:
            while not self.platform.window.should_close:
                self.platform.poll_events()
                # self.renderer.render()

        finally:
            self.close()

    def close(self) -> None:
            """Turn off the application."""
            # self.renderer.close()
            self.platform.close()
