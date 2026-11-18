"""Vulkan rendering implementation."""

from __future__ import annotations
from mosaic.engine.platform.backend.vulkan.vulkan_backend import DrawFrameStatus

from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_backend import DrawFrameStatus
from mosaic.engine.renderer.render_driver import RenderDriver
from mosaic.engine.renderer.vulkan_renderer.triangle_pass import TrianglePass
from mosaic.engine.renderer.vulkan_renderer.vulkan_command_pool import VulkanCommandPool
from mosaic.engine.renderer.vulkan_renderer.vulkan_render_stage import RenderStage
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_backend import VulkanBackend


class VulkanRenderer(RenderDriver):
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
        result = self._backend.draw_frame(self._command_pool.command_buffers)

        if (
            result == DrawFrameStatus.RECREATE_SWAPCHAIN
            and self._backend.wait_for_nonzero_framebuffer()
        ):
            self._recreate_swapchain_resources()


    def resize(self, width: int = 0, height: int = 0) -> None:
        """
        Resize the renderer.

        Args:
            width (int): Width of the rendering area.
            height (int): Height of the rendering area.

        """
        self._backend.wait_idle()

        for render_stage in self._passes:
            render_stage.resize(width, height)

        self._record_command_buffers()

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

    def _recreate_swapchain_resources(self) -> None:
        self._backend.wait_idle()
        self._command_pool.close()
        for render_stage in reversed(self._passes):
            render_stage.close()

        self._backend.recreate_swapchain()
        self._passes = [TrianglePass(self._backend.device, self._backend.swapchain)]
        self._command_pool = VulkanCommandPool(
            self._backend.device, len(self._backend.swapchain.swapchain_images)
        )

        self._record_command_buffers()
