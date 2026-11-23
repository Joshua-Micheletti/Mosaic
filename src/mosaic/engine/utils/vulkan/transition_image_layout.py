"""Vulkan image-layout transition helper."""

from ctypes import byref, pointer
from dataclasses import dataclass

import vulkan_py as vk


@dataclass(frozen=True)
class ImageLayoutTransition:
    """Describe the synchronization and layouts for an image transition."""

    old_layout: int
    new_layout: int
    src_access_mask: int
    dst_access_mask: int
    src_stage_mask: int
    dst_stage_mask: int


def transition_image_layout(
    device: vk.Device,
    command_buffer: vk.VkCommandBuffer,
    image: vk.VkImage,
    transition: ImageLayoutTransition,
) -> None:
    """Record an image layout transition into a command buffer."""
    image_barrier = vk.VkImageMemoryBarrier2(
        srcStageMask=transition.src_stage_mask,
        srcAccessMask=transition.src_access_mask,
        dstStageMask=transition.dst_stage_mask,
        dstAccessMask=transition.dst_access_mask,
        oldLayout=transition.old_layout,
        newLayout=transition.new_layout,
        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        image=image,
        subresourceRange=vk.VkImageSubresourceRange(
            aspectMask=vk.VkImageAspectFlagBits.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    dependency_info = vk.VkDependencyInfo(
        imageMemoryBarrierCount=1,
        pImageMemoryBarriers=pointer(image_barrier),
    )
    device.vkCmdPipelineBarrier2(command_buffer, byref(dependency_info))
