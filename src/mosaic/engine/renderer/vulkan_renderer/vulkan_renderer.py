"""Vulkan rendering implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_backend import DrawFrameStatus
from mosaic.engine.renderer.render_driver import RenderDriver
from mosaic.engine.renderer.vulkan_renderer.triangle_pass import TrianglePass
from mosaic.engine.renderer.vulkan_renderer.vulkan_command_pool import VulkanCommandPool
from mosaic.engine.utils.vulkan.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_backend import VulkanBackend
    from mosaic.engine.renderer.vulkan_renderer.vulkan_render_stage import RenderStage


class VulkanRenderer(RenderDriver):
    """Record configured render stages and ask the backend to present each frame."""

    def __init__(self, backend: VulkanBackend) -> None:
        self._backend = backend
        self._command_pool = VulkanCommandPool(
            backend.device, len(backend.swapchain.swapchain_images)
        )
        self._passes: list[RenderStage] = [
            TrianglePass(
                backend.device,
                backend.swapchain,
                self._command_pool.command_pool,
                self._backend.MAX_FRAMES_IN_FLIGHT,
            )
        ]
        self._record_command_buffers()

    @override
    def run(self) -> None:
        """Render and present one frame."""
        result = self._backend.draw_frame(self._command_pool.command_buffers, self._update_render_stages)

        if result == DrawFrameStatus.RECREATE_SWAPCHAIN:
            self.resize()

    @override
    def close(self) -> None:
        """Release renderer resources while the Vulkan device is alive."""
        self._backend.wait_idle()
        self._command_pool.close()
        for render_stage in reversed(self._passes):
            render_stage.close()

    @override
    def resize(self, width: int = 0, height: int = 0) -> None:
        self._backend.wait_for_framebuffer()
        self._backend.wait_idle()
        self._command_pool.close()

        for render_stage in reversed(self._passes):
            render_stage.close()

        self._backend.recreate_swapchain()
        self._command_pool = VulkanCommandPool(
            self._backend.device, len(self._backend.swapchain.swapchain_images)
        )
        self._passes = [
            TrianglePass(
                self._backend.device, self._backend.swapchain, self._command_pool.command_pool
            )
        ]

        self._record_command_buffers()

    def _record_command_buffers(self, frame_index: int = 0) -> None:
        for image_index, command_buffer in enumerate(self._command_pool.command_buffers):
            check(
                self._backend.device.device.vkBeginCommandBuffer(
                    command_buffer, vk.VkCommandBufferBeginInfo()
                ),
                "vkBeginCommandBuffer",
            )

            for render_stage in self._passes:
                render_stage.record(command_buffer, image_index, frame_index)

            check(
                self._backend.device.device.vkEndCommandBuffer(command_buffer),
                "vkEndCommandBuffer",
            )

    def _update_render_stages(self, frame_index: int) -> None:
        for render_pass in self._passes:
            render_pass.update(frame_index)
