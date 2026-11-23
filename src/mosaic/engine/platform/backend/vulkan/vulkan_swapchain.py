"""Vulkan Swapchain Module."""

import ctypes
from ctypes import byref, c_uint32
from typing import TYPE_CHECKING, Any

import vulkan_py as vk

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.debug import check

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
    from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface

INVALID_EXTENT_SIZE = 0xFFFFFFFF


class VulkanSwapchain(Closable):
    """
    Vulkan Swapchain Class.

    Creates an object that manages images and presenting them to the window.
    """

    _swapchain: vk.VkSwapchainKHR
    _swapchain_format: vk.VkFormat
    _swapchain_extent: vk.VkExtent2D
    _swapchain_images: list[Any]
    _swapchain_image_views: list[vk.VkImageView]
    _device: VulkanDevice

    @property
    def swapchain(self) -> vk.VkSwapchainKHR:
        return self._swapchain

    @property
    def swapchain_format(self) -> vk.VkFormat:
        return self._swapchain_format

    @property
    def swapchain_extent(self) -> vk.VkExtent2D:
        return self._swapchain_extent

    @property
    def swapchain_images(self) -> list[Any]:
        return self._swapchain_images

    @property
    def swapchain_image_views(self) -> list[vk.VkImageView]:
        return self._swapchain_image_views

    def __init__(
        self, width: int, height: int, device: VulkanDevice, surface: VulkanSurface
    ) -> None:
        self._device = device
        self._create_swapchain(width, height, device, surface)
        self._create_image_views(device)

    def close(self) -> None:
        """Close the swapchain."""
        for view in self._swapchain_image_views:
            self._device.device.vkDestroyImageView(view, None)

        self._device.device.vkDestroySwapchainKHR(self._swapchain, None)

    def _create_swapchain(
        self, width: int, height: int, device: VulkanDevice, surface: VulkanSurface
    ) -> None:
        """
        Create the vulkan swapchain object.

        Args:
            width (int): Width.
            height (int): Height.
            window (_Pointer[glfw._GLFWwindow]): GLFW window pointer.
            device (VulkanDevice): Vulkan device.
            surface (VulkanSurface): Vulkan surface.

        """
        # Extract the capabilities of the surface and choose the best format,
        # present mode, and extent for the swapchain.
        caps = vk.VkSurfaceCapabilitiesKHR()
        surface.instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device.physical_device, surface.surface, byref(caps)
        )

        # Extract the formats supported by the surface.
        fmt_count = c_uint32(0)
        surface.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, surface.surface, byref(fmt_count), None
        )
        formats = (vk.VkSurfaceFormatKHR * fmt_count.value)()
        surface.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, surface.surface, byref(fmt_count), formats
        )

        # choose the first format available by default
        chosen_format: vk.VkSurfaceFormatKHR = formats[0]
        # choose a better format if available
        # in this case, B8G8R8A8_UNORM with SRGB nonlinear colorspace
        for f in formats:
            if (
                f.format == vk.VkFormat.VK_FORMAT_B8G8R8A8_UNORM
                and f.colorSpace == vk.VkColorSpaceKHR.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                chosen_format: vk.VkSurfaceFormatKHR = f
                break

        # extract the present modes supported by the surface.
        mode_count = c_uint32(0)
        surface.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, surface.surface, byref(mode_count), None
        )
        modes = (ctypes.c_int32 * mode_count.value)()
        surface.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, surface.surface, byref(mode_count), modes
        )

        # choose FIFO as default present mode.
        # (FIRST IN FIRST OUT queue, frames are presented as they are ready)
        present_mode = vk.VkPresentModeKHR.VK_PRESENT_MODE_FIFO_KHR  # always guaranteed by the spec
        # if MAILBOX is available, choose it as the present mode.
        # (Skips frames if the queue is full, but always presents the most recent frame)
        if vk.VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR in list(modes):
            present_mode = vk.VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR

        # if the extent is valid, use it, otherwise, use the window size as the extent.
        if caps.currentExtent.width != INVALID_EXTENT_SIZE:
            extent: vk.VkExtent2D = caps.currentExtent
        else:
            extent: vk.VkExtent2D = vk.VkExtent2D(width=width, height=height)

        # determine the number of images in the swapchain.
        image_count = caps.minImageCount + 1
        if caps.maxImageCount != 0 and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

        # if the graphics and present (queue) families are different, use concurrent sharing mode
        # otherwise, use exclusive sharing mode.
        if device.graphics_family != device.present_family:
            indices = (c_uint32 * 2)(device.graphics_family, device.present_family)
            sharing_mode, index_count, indices_ptr = (
                vk.VkSharingMode.VK_SHARING_MODE_CONCURRENT,
                2,
                indices,
            )
        else:
            sharing_mode, index_count, indices_ptr = (
                vk.VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
                0,
                None,
            )

        # create the swapchain with the chosen parameters.
        create_info = vk.VkSwapchainCreateInfoKHR(
            surface=surface.surface,
            minImageCount=image_count,
            imageFormat=chosen_format.format,
            imageColorSpace=chosen_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VkImageUsageFlagBits.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=sharing_mode,
            queueFamilyIndexCount=index_count,
            pQueueFamilyIndices=indices_ptr,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VkCompositeAlphaFlagBitsKHR.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=1,
            oldSwapchain=vk.VkSwapchainKHR(0),
        )

        swapchain = vk.VkSwapchainKHR()
        check(
            device.device.vkCreateSwapchainKHR(create_info, None, byref(swapchain)),
            "vkCreateSwapchainKHR",
        )
        self._swapchain, self._swapchain_format, self._swapchain_extent = (
            swapchain,
            chosen_format.format,
            extent,
        )

        # extract the swapchain images from the swapchain object.
        img_count = c_uint32(0)
        device.device.vkGetSwapchainImagesKHR(self._swapchain, byref(img_count), None)
        images = (vk.VkImage * img_count.value)()
        device.device.vkGetSwapchainImagesKHR(self._swapchain, byref(img_count), images)
        self._swapchain_images = list(images)

    def _create_image_views(self, device: VulkanDevice) -> None:
        """
        Create the image views of the swapchain images.

        Args:
            device (VulkanDevice): Vulkan Device.

        """
        self._swapchain_image_views = []
        for img in self._swapchain_images:
            create_info = vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VkImageViewType.VK_IMAGE_VIEW_TYPE_2D,
                format=self._swapchain_format,
                components=vk.VkComponentMapping(
                    r=vk.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VkImageAspectFlagBits.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            view = vk.VkImageView()
            check(
                device.device.vkCreateImageView(create_info, None, byref(view)), "vkCreateImageView"
            )
            self._swapchain_image_views.append(view)
