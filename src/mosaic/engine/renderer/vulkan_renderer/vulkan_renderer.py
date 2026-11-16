"""Vulkan rendering implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import vulkan_py as vk

from mosaic.engine.renderer.vulkan_renderer.triangle_pass import TrianglePass
from mosaic.engine.renderer.vulkan_renderer.vulkan_command_pool import VulkanCommandPool
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.lifecycle.runnable import Runnable
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_backend import VulkanBackend


class RenderStage(Protocol):
    """A Vulkan render stage that records work into a command buffer."""

    def record(self, command_buffer: vk.VkCommandBuffer, image_index: int) -> None: ...

    def close(self) -> None: ...


class VulkanRenderer(Closable, Runnable):
    """Record configured render stages and ask the backend to present each frame."""

    def __init__(self, backend: VulkanBackend) -> None:
        self._backend = backend
        self._passes: list[RenderStage] = [TrianglePass(backend.device, backend.swapchain)]
        self._command_pool = VulkanCommandPool(
            backend.device, len(backend.swapchain.swapchain_images)
        )
        self._record_command_buffers()

    def run(self) -> None:
        """Render and present one frame."""
        self._backend.draw_frame(self._command_pool.command_buffers)

    def close(self) -> None:
        """Release renderer resources while the Vulkan device is alive."""
        self._backend.wait_idle()
        self._command_pool.close()
        for render_stage in reversed(self._passes):
            render_stage.close()

    def _record_command_buffers(self) -> None:
        for image_index, command_buffer in enumerate(self._command_pool.command_buffers):
            check(
                self._backend.device.device.vkBeginCommandBuffer(
                    command_buffer, vk.VkCommandBufferBeginInfo()
                ),
                "vkBeginCommandBuffer",
            )
            for render_stage in self._passes:
                render_stage.record(command_buffer, image_index)
            check(
                self._backend.device.device.vkEndCommandBuffer(command_buffer),
                "vkEndCommandBuffer",
            )
