import ctypes
from ctypes import _Pointer, byref, c_uint32, pointer
from typing import Any

import glfw
import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check


class VulkanSwapchain(Closable):
    _swapchain: vk.VkSwapchainKHR
    _swapchain_format: Any
    _swapchain_extent: Any | vk.VkExtent2D
    _swapchain_images: list[Any]
    _swapchain_image_views: list[vk.VkImageView]
    _render_pass: vk.VkRenderPass
    _device: VulkanDevice

    @property
    def swapchain(self) -> vk.VkSwapchainKHR:
        return self._swapchain

    @property
    def swapchain_format(self) -> Any:
        return self._swapchain_format

    @property
    def swapchain_extent(self) -> Any | vk.VkExtent2D:
        return self._swapchain_extent

    @property
    def swapchain_images(self) -> list[Any]:
        return self._swapchain_images

    @property
    def swapchain_image_views(self) -> list[vk.VkImageView]:
        return self._swapchain_image_views

    @property
    def render_pass(self) -> vk.VkRenderPass:
        return self._render_pass

    def __init__(
        self, window: _Pointer[glfw._GLFWwindow], device: VulkanDevice, surface: VulkanSurface
    ) -> None:
        self._device = device
        self._create_swapchain(window, device, surface)
        self._create_image_views(device)
        self._create_render_pass(device)

    def close(self) -> None:
        """Close the swapchain."""
        self._device.device.vkDestroyRenderPass(self._render_pass, None)

        for view in self._swapchain_image_views:
            self._device.device.vkDestroyImageView(view, None)

        self._device.device.vkDestroySwapchainKHR(self._swapchain, None)

    def _create_swapchain(
        self, window: _Pointer[glfw._GLFWwindow], device: VulkanDevice, surface: VulkanSurface
    ) -> None:
        caps = vk.VkSurfaceCapabilitiesKHR()
        surface.instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device.physical_device, surface.surface, byref(caps)
        )

        fmt_count = c_uint32(0)
        surface.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, surface.surface, byref(fmt_count), None
        )
        formats = (vk.VkSurfaceFormatKHR * fmt_count.value)()
        surface.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, surface.surface, byref(fmt_count), formats
        )
        chosen_format = formats[0]
        for f in formats:
            if (
                f.format == vk.VkFormat.VK_FORMAT_B8G8R8A8_UNORM
                and f.colorSpace == vk.VkColorSpaceKHR.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                chosen_format = f
                break

        mode_count = c_uint32(0)
        surface.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, surface.surface, byref(mode_count), None
        )
        modes = (ctypes.c_int32 * mode_count.value)()
        surface.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, surface.surface, byref(mode_count), modes
        )
        present_mode = vk.VkPresentModeKHR.VK_PRESENT_MODE_FIFO_KHR  # always guaranteed by the spec
        if vk.VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR in list(modes):
            present_mode = vk.VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR

        if caps.currentExtent.width != 0xFFFFFFFF:
            extent = caps.currentExtent
        else:
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            extent = vk.VkExtent2D(width=fb_w, height=fb_h)

        image_count = caps.minImageCount + 1
        if caps.maxImageCount != 0 and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

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

        img_count = c_uint32(0)
        device.device.vkGetSwapchainImagesKHR(self._swapchain, byref(img_count), None)
        images = (vk.VkImage * img_count.value)()
        device.device.vkGetSwapchainImagesKHR(self._swapchain, byref(img_count), images)
        self._swapchain_images = list(images)

    def _create_image_views(self, device: VulkanDevice) -> None:
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

    def _create_render_pass(self, device: VulkanDevice) -> None:
        color_attachment = vk.VkAttachmentDescription(
            format=self._swapchain_format,
            samples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0, layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=pointer(color_ref),
        )
        dependency = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstAccessMask=vk.VkAccessFlagBits.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )

        create_info = vk.VkRenderPassCreateInfo(
            attachmentCount=1,
            pAttachments=pointer(color_attachment),
            subpassCount=1,
            pSubpasses=pointer(subpass),
            dependencyCount=1,
            pDependencies=pointer(dependency),
        )
        render_pass = vk.VkRenderPass()
        check(
            device.device.vkCreateRenderPass(create_info, None, byref(render_pass)),
            "vkCreateRenderPass",
        )
        self._render_pass = render_pass
