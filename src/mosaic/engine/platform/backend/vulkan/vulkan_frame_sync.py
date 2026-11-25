"""Vulkan per-frame synchronization resources."""

from ctypes import byref
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
    from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain


class VulkanFrameSync(Closable):
    """Manage semaphores and fences used by the backend frame loop."""

    _device: VulkanDevice
    _image_available_semaphores: list[vk.VkSemaphore]
    _in_flight_fences: list[vk.VkFence]
    _render_finished_semaphores: list[vk.VkSemaphore]
    _current_frame: int
    _MAX_FRAMES_IN_FLIGHT: int

    @property
    def image_available_semaphores(self) -> list[vk.VkSemaphore]:
        return self._image_available_semaphores

    @property
    def in_flight_fences(self) -> list[vk.VkFence]:
        return self._in_flight_fences

    @property
    def render_finished_semaphores(self) -> list[vk.VkSemaphore]:
        return self._render_finished_semaphores

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def __init__(
        self, device: VulkanDevice, swapchain: VulkanSwapchain, max_frames_in_flight: int = 2
    ) -> None:
        self._device = device
        self._MAX_FRAMES_IN_FLIGHT = max_frames_in_flight
        self._create_sync_objects(swapchain)

    def advance_frame(self) -> None:
        """Advance the frame slot."""
        self._current_frame = (self._current_frame + 1) % self._MAX_FRAMES_IN_FLIGHT

    def close(self) -> None:
        """Destroy synchronization resources."""
        for semaphore in self._image_available_semaphores:
            self._device.device.vkDestroySemaphore(semaphore, None)
        for semaphore in self._render_finished_semaphores:
            self._device.device.vkDestroySemaphore(semaphore, None)
        for fence in self._in_flight_fences:
            self._device.device.vkDestroyFence(fence, None)

    def _create_sync_objects(self, swapchain: VulkanSwapchain) -> None:
        self._image_available_semaphores = []
        self._in_flight_fences = []
        semaphore_info = vk.VkSemaphoreCreateInfo()
        fence_info = vk.VkFenceCreateInfo(
            flags=vk.VkFenceCreateFlagBits.VK_FENCE_CREATE_SIGNALED_BIT
        )

        for _ in range(self._MAX_FRAMES_IN_FLIGHT):
            image_available = vk.VkSemaphore()
            fence = vk.VkFence()
            check(
                self._device.device.vkCreateSemaphore(semaphore_info, None, byref(image_available)),
                "vkCreateSemaphore",
            )
            check(
                self._device.device.vkCreateFence(fence_info, None, byref(fence)),
                "vkCreateFence",
            )
            self._image_available_semaphores.append(image_available)
            self._in_flight_fences.append(fence)

        self._render_finished_semaphores = []
        for _ in swapchain.swapchain_images:
            semaphore = vk.VkSemaphore()
            check(
                self._device.device.vkCreateSemaphore(semaphore_info, None, byref(semaphore)),
                "vkCreateSemaphore",
            )
            self._render_finished_semaphores.append(semaphore)

        self._current_frame = 0
