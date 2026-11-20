"""Vulkan memory utilities."""

import vulkan_py as vk


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

    raise RuntimeError("failed to find suitable memory type!")
