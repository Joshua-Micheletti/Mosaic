import ctypes
from ctypes import _Pointer, byref, c_uint32, pointer
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_frame_sync import VulkanFrameSync
from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check

if TYPE_CHECKING:
    import glfw

UINT64_MAX = 0xFFFFFFFFFFFFFFFF


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

    @property
    def swapchain(self) -> VulkanSwapchain:
        return self._swapchain

    def __init__(self, window: _Pointer[glfw._GLFWwindow]) -> None:
        self._surface = VulkanSurface(window)
        self._device = VulkanDevice(self._surface.instance, self._surface.surface)
        self._swapchain = VulkanSwapchain(window, self._device, self._surface)
        self._frame_sync = VulkanFrameSync(self._device, self._swapchain)
        # https://github.com/StormMGamming41/vulkan-py/blob/main/examples/hello_triangle.py#L21

    def close(self) -> None:
        """Closing routine."""
        self.wait_idle()
        self._frame_sync.close()
        self._swapchain.close()
        self._device.close()
        self._surface.close()

    def wait_idle(self) -> None:
        """Wait for all device work to finish."""
        check(self._device.device.vkDeviceWaitIdle(), "vkDeviceWaitIdle")

    def draw_frame(self, command_buffers: list[vk.VkCommandBuffer]) -> None:
        """Acquire, submit, and present a frame using renderer-recorded commands."""
        frame = self._frame_sync.current_frame
        fence = self._frame_sync.in_flight_fences[frame]
        self._device.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)
        self._device.device.vkResetFences(1, byref(fence))

        image_index = c_uint32(0)
        self._device.device.vkAcquireNextImageKHR(
            self._swapchain.swapchain,
            UINT64_MAX,
            self._frame_sync.image_available_semaphores[frame],
            vk.VkFence(0),
            byref(image_index),
        )

        wait_semaphores = (vk.VkSemaphore * 1)(self._frame_sync.image_available_semaphores[frame])
        signal_semaphores = (vk.VkSemaphore * 1)(
            self._frame_sync.render_finished_semaphores[image_index.value]
        )
        wait_stages = (ctypes.c_uint32 * 1)(
            vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        cmd_buffers = (vk.VkCommandBuffer * 1)(command_buffers[image_index.value])

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

        self._frame_sync.advance_frame()
