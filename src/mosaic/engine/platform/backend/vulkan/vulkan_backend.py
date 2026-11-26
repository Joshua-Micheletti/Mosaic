"""Vulkan Backend Module."""

import ctypes
from ctypes import byref, c_uint32, pointer
from enum import IntEnum
from typing import TYPE_CHECKING, Callable

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_frame_sync import VulkanFrameSync
from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.window.window import Window

UINT64_MAX = 0xFFFFFFFFFFFFFFFF

class DrawFrameStatus(IntEnum):
    """Enumerator for the returning status of the rendering step."""

    OK = 0
    RECREATE_SWAPCHAIN = 1


class VulkanBackend(Closable):
    """Vulkan Backend class."""

    _device: VulkanDevice
    _surface: VulkanSurface
    _window: Window
    _MAX_FRAMES_IN_FLIGHT: int

    @property
    def device(self) -> VulkanDevice:
        return self._device

    @property
    def surface(self) -> VulkanSurface:
        return self._surface

    @property
    def swapchain(self) -> VulkanSwapchain:
        return self._swapchain

    @property
    def MAX_FRAMES_IN_FLIGHT(self) -> int:
        return self._MAX_FRAMES_IN_FLIGHT 

    @property
    def current_frame(self) -> int:
        return self._frame_sync.current_frame

    def __init__(self, window: Window) -> None:
        self._MAX_FRAMES_IN_FLIGHT = 2

        self._window = window
        self._surface = VulkanSurface(window)
        self._device = VulkanDevice(self._surface.instance, self._surface.surface)
        self._swapchain = VulkanSwapchain(window.width, window.height, self._device, self._surface)
        self._frame_sync = VulkanFrameSync(self._device, self._swapchain, self._MAX_FRAMES_IN_FLIGHT)
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

    def draw_frame(self, command_buffers: list[vk.VkCommandBuffer], update_frame: Callable[[int], None]) -> DrawFrameStatus:
        """Acquire, submit, and present a frame using renderer-recorded commands."""
        frame = self._frame_sync.current_frame
        fence = self._frame_sync.in_flight_fences[frame]
        fence_result = self._device.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)

        check(fence_result, "vkWaitForFences")

        # acquire the next image from the swapchain
        image_index = c_uint32(0)
        result = self._device.device.vkAcquireNextImageKHR(
            self._swapchain.swapchain,
            UINT64_MAX,  # disable timeout, wait indefinitely
            self._frame_sync.image_available_semaphores[frame],  # signal semaphore
            vk.VkFence(0),  # no fence, we use the fence above
            byref(image_index),  # output parameter for the acquired image index
        )

        if result == vk.VkResult.VK_ERROR_OUT_OF_DATE_KHR:
            return DrawFrameStatus.RECREATE_SWAPCHAIN

        acquired_suboptimal = result == vk.VkResult.VK_SUBOPTIMAL_KHR
        if result != vk.VkResult.VK_SUCCESS and not acquired_suboptimal:
            check(result, "vkAcquireNextImageKHR")

        update_frame(image_index.value)

        self._device.device.vkResetFences(1, byref(fence))

        # semaphores that signal when the swapchain image is available
        wait_semaphores = (vk.VkSemaphore * 1)(self._frame_sync.image_available_semaphores[frame])
        # semaphores that signal when the rendering is finished
        signal_semaphores = (vk.VkSemaphore * 1)(
            self._frame_sync.render_finished_semaphores[image_index.value]
        )
        # wait for the color attachment to be filled
        wait_stages = (ctypes.c_uint32 * 1)(
            vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        # pick the command buffer to bind and execute
        cmd_buffers = (vk.VkCommandBuffer * 1)(command_buffers[image_index.value])

        # compile the submit info with the synchronization steps and the command buffer to execute
        submit_info = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=1,
            pSignalSemaphores=signal_semaphores,
        )
        # submit the info to the queue
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
        present_result = self._device.device.vkQueuePresentKHR(
            self._device.present_queue, byref(present_info)
        )

        self._frame_sync.advance_frame()

        if acquired_suboptimal or present_result in (
            vk.VkResult.VK_SUBOPTIMAL_KHR,
            vk.VkResult.VK_ERROR_OUT_OF_DATE_KHR,
        ):
            return DrawFrameStatus.RECREATE_SWAPCHAIN

        check(present_result, "vkQueuePresentKHR")

        return DrawFrameStatus.OK

    def recreate_swapchain(self) -> None:
        """Recreate swapchain resources after the renderer releases its dependents."""
        self._frame_sync.close()
        self._swapchain.close()
        size = self._window.size
        self._swapchain = VulkanSwapchain(size[0], size[1], self._device, self._surface)
        self._frame_sync = VulkanFrameSync(self._device, self._swapchain)

    def wait_for_framebuffer(self) -> bool:
        """
        Wrapper for the window wait for framebuffer helper.

        Returns:
            bool: true when the framebuffer is available

        """
        return self._window.wait_for_framebuffer()


