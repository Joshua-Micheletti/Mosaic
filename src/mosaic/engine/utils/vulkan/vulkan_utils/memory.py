"""Vulkan memory utilities."""

import ctypes
from ctypes import byref, c_void_p
from typing import Any

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check


def find_memory_type(
    type_filter: int,
    properties: int,
    memory_properties: vk.VkPhysicalDeviceMemoryProperties,
) -> int:
    """Find a physical-device memory type matching the filter and required properties."""
    for index in range(memory_properties.memoryTypeCount):
        if (type_filter & (1 << index)) and (
            memory_properties.memoryTypes[index].propertyFlags & properties
        ) == properties:
            return index

    error = "failed to find suitable memory type!"
    raise RuntimeError(error)


def create_buffer(
    data_length: int,
    device: VulkanDevice,
    usage: int,
    properties: int,
) -> tuple[vk.VkBuffer, vk.VkDeviceMemory]:
    buffer_info = vk.VkBufferCreateInfo(
        size=data_length,
        usage=usage,
        # usage=vk.VkBufferUsageFlagBits.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        sharingMode=vk.VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
    )

    buffer = vk.VkBuffer()
    check(
        device.device.vkCreateBuffer(buffer_info, None, byref(buffer)),
        "vkCreateBuffer",
    )

    memory_requirements = vk.VkMemoryRequirements()

    device.device.vkGetBufferMemoryRequirements(buffer, byref(memory_requirements))

    memory_properties = device.memory_properties

    memory_allocate_info = vk.VkMemoryAllocateInfo(
        allocationSize=memory_requirements.size,
        memoryTypeIndex=find_memory_type(
            type_filter=memory_requirements.memoryTypeBits,
            properties=properties,
            # properties=vk.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            #| vk.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory_properties=memory_properties,
        ),
    )

    buffer_memory = vk.VkDeviceMemory()
    check(
        device.device.vkAllocateMemory(
            memory_allocate_info,
            None,
            byref(buffer_memory),
        ),
        "vkAllocateMemory",
    )

    check(
        device.device.vkBindBufferMemory(
            buffer,
            buffer_memory,
            0,
        ),
        "vkBindBufferMemory",
    )

    return (buffer, buffer_memory)


def load_data_to_buffer(data: Any, device: VulkanDevice, buffer_memory: vk.VkDeviceMemory) -> None:
    mapped = c_void_p()
    size = len(data)

    check(
        device.device.vkMapMemory(buffer_memory, 0, size, 0, byref(mapped)),
        "vkMapMemory",
    )

    ctypes.memmove(mapped, data, size)
    device.device.vkUnmapMemory(buffer_memory)


def copy_buffer(
    source: vk.VkBuffer,
    destination: vk.VkBuffer,
    size: int,
    device: VulkanDevice,
    command_pool: vk.VkCommandPool,
) -> None:
    """
    Copy the bytes from a source buffer to a destination buffer.

    Works by creating a command buffer with the operation of copying the data
    from one buffer to the other, submitting the work to the GPU and cleaning up

    Args:
        source (vk.VkBuffer): Source buffer.
        destination (vk.VkBuffer): Destination buffer.
        size (int): Size of the data to move.
        device (VulkanDevice): Vulkan Device.
        command_pool (vk.VkCommandPool): Command Pool.

    """
    # define the region of data to copy
    copy_region = vk.VkBufferCopy(
        srcOffset=0,
        dstOffset=0,
        size=size,
    )
    copy_regions = (vk.VkBufferCopy * 1)(copy_region)

    # allocate a command buffer for the moving operation
    allocate_info = vk.VkCommandBufferAllocateInfo(
        commandPool=command_pool,
        level=vk.VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    command_buffers = (vk.VkCommandBuffer * 1)()
    check(
        device.device.vkAllocateCommandBuffers(allocate_info, command_buffers),
        "vkAllocateCommandBuffers",
    )
    command_buffer = command_buffers[0]

    # define the command buffer to be a one time use
    begin_info = vk.VkCommandBufferBeginInfo(
        flags=vk.VkCommandBufferUsageFlagBits.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    )

    check(
        device.device.vkBeginCommandBuffer(command_buffer, begin_info),
        "vkBeginCommandBuffer",
    )

    # send the command to copy the data between buffers
    device.device.vkCmdCopyBuffer(
        command_buffer,
        source,
        destination,
        1,
        copy_regions,
    )

    # end the operations in the command buffer
    check(
        device.device.vkEndCommandBuffer(command_buffer),
        "vkEndCommandBuffer",
    )

    # submit the request of the command buffer to the graphics_queue
    submit_info = vk.VkSubmitInfo(
        commandBufferCount=1,
        pCommandBuffers=command_buffers,
    )

    check(
        device.device.vkQueueSubmit(
            device.graphics_queue,
            1,
            byref(submit_info),
            vk.VkFence(0),
        ),
        "vkQueueSubmit",
    )

    # wait for the device to be idle and free the temporary command buffer
    device.device.vkQueueWaitIdle(device.graphics_queue)
    device.device.vkFreeCommandBuffers(command_pool, 1, command_buffers)
