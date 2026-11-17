"""Engine Module."""

from mosaic.engine.platform.platform import Platform
from mosaic.engine.renderer.renderer import Renderer
from mosaic.engine.renderer.vulkan_renderer.vulkan_renderer import VulkanRenderer
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.runnable import Runnable


class Engine(Closable, Runnable):
    """Manage the application lifecycle and render loop."""

    def __init__(self) -> None:
        self._platform = Platform()
        self._renderer = Renderer(driver=VulkanRenderer(self._platform.backend))

    def run(self) -> None:
        """Run the render loop until the window is closed."""
        while not self._platform.window.should_close:
            self._platform.poll_events()
            self._renderer.run()

    def close(self) -> None:
        """Close the engine and clean up resources."""
        self._renderer.close()
        self._platform.close()
