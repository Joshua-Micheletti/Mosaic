"""Vulkan image-layout transition helper."""

from ctypes import byref, pointer

import vulkan_py as vk


def transition_image_layout(
    device: vk.Device,
    command_buffer: vk.VkCommandBuffer,
    image: vk.VkImage,
    old_layout: int,
    new_layout: int,
    src_access_mask: int,
    dst_access_mask: int,
    src_stage_mask: int,
    dst_stage_mask: int,
) -> None:
    """Record an image layout transition into a command buffer."""
    image_barrier = vk.VkImageMemoryBarrier2(
        srcStageMask=src_stage_mask,
        srcAccessMask=src_access_mask,
        dstStageMask=dst_stage_mask,
        dstAccessMask=dst_access_mask,
        oldLayout=old_layout,
        newLayout=new_layout,
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
