"""Vulkan rendering stage."""

from abc import abstractmethod
from typing import TYPE_CHECKING

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.resizable import Resizable

if TYPE_CHECKING:
    import vulkan_py as vk


class RenderStage(Closable, Resizable):
    """A Vulkan render stage that records work into a command buffer."""

    @abstractmethod
    def record(self, command_buffer: vk.VkCommandBuffer, image_index: int) -> None:
        """
        Record the render stage.

        Args:
            command_buffer (vk.VkCommandBuffer): Command buffer to record into.
            image_index (int): Index of the image to render to.

        """
