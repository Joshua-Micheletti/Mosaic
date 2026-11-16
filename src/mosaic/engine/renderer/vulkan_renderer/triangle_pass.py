"""Vulkan pass that owns and draws the starter triangle."""

from __future__ import annotations

from ctypes import byref, pointer
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check
from mosaic.engine.utils.vulkan.vulkan_utils.shader import load_shader_module

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
    from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain


class TrianglePass(Closable):
    """Create the triangle's Vulkan resources and record its draw."""

    _render_pass: vk.VkRenderPass
    _pipeline_layout: vk.VkPipelineLayout
    _pipeline: vk.VkPipeline
    _framebuffers: list[vk.VkFramebuffer]

    def __init__(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        self._device = device
        self._swapchain = swapchain
        self._create_render_pass()
        self._create_graphics_pipeline()
        self._create_framebuffers()

    def record(self, command_buffer: vk.VkCommandBuffer, image_index: int) -> None:
        """Record the triangle draw for one swapchain image."""
        clear_value = vk.VkClearValue(color=vk.VkClearColorValue(float32=(0.01, 0.01, 0.02, 1.0)))
        render_pass_info = vk.VkRenderPassBeginInfo(
            renderPass=self._render_pass,
            framebuffer=self._framebuffers[image_index],
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=self._swapchain.swapchain_extent,
            ),
            clearValueCount=1,
            pClearValues=pointer(clear_value),
        )
        self._device.device.vkCmdBeginRenderPass(
            command_buffer,
            byref(render_pass_info),
            vk.VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE,
        )
        self._device.device.vkCmdBindPipeline(
            command_buffer,
            vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._pipeline,
        )
        self._device.device.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self._device.device.vkCmdEndRenderPass(command_buffer)

    def close(self) -> None:
        """Release resources owned by this pass."""
        for framebuffer in self._framebuffers:
            self._device.device.vkDestroyFramebuffer(framebuffer, None)

        self._device.device.vkDestroyPipeline(self._pipeline, None)
        self._device.device.vkDestroyPipelineLayout(self._pipeline_layout, None)
        self._device.device.vkDestroyRenderPass(self._render_pass, None)

    def _create_render_pass(self) -> None:
        color_attachment = vk.VkAttachmentDescription(
            format=self._swapchain.swapchain_format,
            samples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
            self._device.device.vkCreateRenderPass(create_info, None, byref(render_pass)),
            "vkCreateRenderPass",
        )
        self._render_pass = render_pass

    def _create_graphics_pipeline(self) -> None:
        # load the shaders
        vert_module = load_shader_module(
            self._device.device, "./src/mosaic/shaders/triangle.vert.spv"
        )
        frag_module = load_shader_module(
            self._device.device, "./src/mosaic/shaders/triangle.frag.spv"
        )

        # set the shaders in the pipeline stage
        vert_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            module=vert_module,
            pName=b"main",
        )
        frag_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=frag_module,
            pName=b"main",
        )

        # store the stages in an array for the pipeline creation info
        stages = (vk.VkPipelineShaderStageCreateInfo * 2)(vert_stage, frag_stage)

        vertex_input = vk.VkPipelineVertexInputStateCreateInfo()
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self._swapchain.swapchain_extent.width),
            height=float(self._swapchain.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=self._swapchain.swapchain_extent,
        )
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1,
            pViewports=pointer(viewport),
            scissorCount=1,
            pScissors=pointer(scissor),
        )
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            polygonMode=vk.VkPolygonMode.VK_POLYGON_MODE_FILL,
            cullMode=vk.VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            lineWidth=1.0,
        )
        multisample = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT
        )
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=0,
            colorWriteMask=0xF,
        )
        color_blend = vk.VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1,
            pAttachments=pointer(color_blend_attachment),
        )

        pipeline_layout = vk.VkPipelineLayout()
        check(
            self._device.device.vkCreatePipelineLayout(
                vk.VkPipelineLayoutCreateInfo(), None, byref(pipeline_layout)
            ),
            "vkCreatePipelineLayout",
        )
        self._pipeline_layout = pipeline_layout

        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2,
            pStages=stages,
            pVertexInputState=pointer(vertex_input),
            pInputAssemblyState=pointer(input_assembly),
            pViewportState=pointer(viewport_state),
            pRasterizationState=pointer(rasterizer),
            pMultisampleState=pointer(multisample),
            pColorBlendState=pointer(color_blend),
            layout=pipeline_layout,
            renderPass=self._render_pass,
            subpass=0,
        )
        pipeline = vk.VkPipeline()
        check(
            self._device.device.vkCreateGraphicsPipelines(
                vk.VkPipelineCache(0), 1, byref(pipeline_info), None, byref(pipeline)
            ),
            "vkCreateGraphicsPipelines",
        )
        self._pipeline = pipeline

        self._device.device.vkDestroyShaderModule(vert_module, None)
        self._device.device.vkDestroyShaderModule(frag_module, None)

    def _create_framebuffers(self) -> None:
        self._framebuffers = []
        for view in self._swapchain.swapchain_image_views:
            create_info = vk.VkFramebufferCreateInfo(
                renderPass=self._render_pass,
                attachmentCount=1,
                pAttachments=pointer(view),
                width=self._swapchain.swapchain_extent.width,
                height=self._swapchain.swapchain_extent.height,
                layers=1,
            )
            framebuffer = vk.VkFramebuffer()
            check(
                self._device.device.vkCreateFramebuffer(create_info, None, byref(framebuffer)),
                "vkCreateFramebuffer",
            )
            self._framebuffers.append(framebuffer)
