from ctypes import byref
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice


class VulkanCommandPool(Closable):
    """Allocate and own command buffers for renderer work."""

    _command_pool: vk.VkCommandPool
    _command_buffers: list[vk.VkCommandBuffer]
    _device: VulkanDevice

    @property
    def command_pool(self) -> vk.VkCommandPool:
        return self._command_pool

    @property
    def command_buffers(self) -> list[vk.VkCommandBuffer]:
        return self._command_buffers

    def __init__(self, device: VulkanDevice, command_buffer_count: int) -> None:
        self._device = device
        self._create_command_pool(device)
        self._create_command_buffers(device, command_buffer_count)

    def close(self) -> None:
        """Close the command pool."""
        self._device.device.vkDestroyCommandPool(self._command_pool, None)

    def _create_command_pool(self, device: VulkanDevice) -> None:
        # create the command pool for the graphics queue family
        create_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=device.graphics_family,
        )
        pool = vk.VkCommandPool()
        check(
            device.device.vkCreateCommandPool(create_info, None, byref(pool)), "vkCreateCommandPool"
        )
        self._command_pool = pool

    def _create_command_buffers(self, device: VulkanDevice, command_buffer_count: int) -> None:
        # allocate the command buffers from the command pool
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self._command_pool,
            level=vk.VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=command_buffer_count,
        )
        buffers = (vk.VkCommandBuffer * command_buffer_count)()
        check(
            device.device.vkAllocateCommandBuffers(alloc_info, buffers), "vkAllocateCommandBuffers"
        )
        self._command_buffers = list(buffers)
