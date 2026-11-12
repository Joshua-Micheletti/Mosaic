import ctypes
from ctypes import _Pointer, byref, c_uint32, pointer
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_command_pool import VulkanCommandPool
from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_graphics_pipeline import VulkanGraphicsPipeline
from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check

if TYPE_CHECKING:
    import glfw

UINT64_MAX = 0xFFFFFFFFFFFFFFFF
MAX_FRAMES_IN_FLIGHT = 2


class VulkanBackend(Closable):
    """Vulkan Backend class."""

    _device: VulkanDevice
    _surface: VulkanSurface

    @property
    def device(self) -> VulkanDevice:
        return self._device

    @property
    def surface(self) -> VulkanSurface:
        return self._surface

    def __init__(self, window: _Pointer[glfw._GLFWwindow]) -> None:
        self._surface = VulkanSurface(window)
        self._device = VulkanDevice(self._surface.instance, self._surface.surface)
        self._swapchain = VulkanSwapchain(window, self._device, self._surface)
        self._graphics_pipeline = VulkanGraphicsPipeline(self._device, self._swapchain)
        self._command_pool = VulkanCommandPool(
            self._device, self._swapchain, self._graphics_pipeline
        )
        # https://github.com/StormMGamming41/vulkan-py/blob/main/examples/hello_triangle.py#L21

    def close(self) -> None:
        """Closing routine."""
        self._command_pool.close()
        self._graphics_pipeline.close()
        self._swapchain.close()
        self._device.close()
        self._surface.close()

    def draw_frame(self):
        fence = self._command_pool.in_flight_fences[self._command_pool.current_frame]
        self._device.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)
        self._device.device.vkResetFences(1, byref(fence))

        image_index = c_uint32(0)
        self._device.device.vkAcquireNextImageKHR(
            self._swapchain.swapchain,
            UINT64_MAX,
            self._command_pool.image_available_semaphores[self._command_pool.current_frame],
            vk.VkFence(0),
            byref(image_index),
        )

        wait_semaphores = (vk.VkSemaphore * 1)(
            self._command_pool.image_available_semaphores[self._command_pool.current_frame]
        )
        signal_semaphores = (vk.VkSemaphore * 1)(
            self._command_pool.render_finished_semaphores[image_index.value]
        )  # <-- changed
        wait_stages = (ctypes.c_uint32 * 1)(
            vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        cmd_buffers = (vk.VkCommandBuffer * 1)(
            self._command_pool.command_buffers[image_index.value]
        )

        submit_info = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=1,
            pSignalSemaphores=signal_semaphores,
        )
        check(
            self._device.device.vkQueueSubmit(
                self._device.graphics_queue, 1, byref(submit_info), fence
            ),
            "vkQueueSubmit",
        )

        swapchains = (vk.VkSwapchainKHR * 1)(self._swapchain.swapchain)
        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=swapchains,
            pImageIndices=pointer(image_index),
        )
        self._device.device.vkQueuePresentKHR(self._device.present_queue, byref(present_info))

        self._command_pool.advance_frame()
