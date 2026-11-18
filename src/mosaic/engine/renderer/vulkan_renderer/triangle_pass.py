"""Vulkan pass that owns and draws the starter triangle."""

from __future__ import annotations

from ctypes import byref, c_int32, c_void_p, cast, pointer
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.renderer.vulkan_renderer.vulkan_render_stage import RenderStage
from mosaic.engine.utils.vulkan.vulkan_utils.debug import check
from mosaic.engine.utils.vulkan.vulkan_utils.shader import load_shader_module
from mosaic.engine.utils.vulkan.vulkan_utils.transition_image_layout import transition_image_layout

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
    from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain


class TrianglePass(RenderStage):
    """Create the triangle's Vulkan resources and record its draw."""

    _render_pass: vk.VkRenderPass
    _pipeline_layout: vk.VkPipelineLayout
    _pipeline: vk.VkPipeline
    _framebuffers: list[vk.VkFramebuffer]
    _viewport: vk.VkViewport
    _scissor: vk.VkRect2D

    def __init__(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        self._device = device
        self._swapchain = swapchain
        self._create_graphics_pipeline()

    def record(self, command_buffer: vk.VkCommandBuffer, image_index: int) -> None:
        """Record the triangle draw for one swapchain image."""
        # Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
        transition_image_layout(
            device=self._device.device,
            command_buffer=command_buffer,
            image=self._swapchain.swapchain_images[image_index],
            old_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            new_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_NONE,
            dst_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            src_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            dst_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        )

        clear_value = vk.VkClearValue(color=vk.VkClearColorValue(float32=(0.01, 0.01, 0.02, 1.0)))

        rendering_attachment_info = vk.VkRenderingAttachmentInfo(
            imageView=self._swapchain.swapchain_image_views[image_index],
            imageLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            loadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            clearValue=clear_value,
        )

        rendering_info = vk.VkRenderingInfo(
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0), extent=self._swapchain.swapchain_extent
            ),
            layerCount=1,
            colorAttachmentCount=1,
            pColorAttachments=pointer(rendering_attachment_info),
        )

        # define the rendering commands
        self._device.device.vkCmdBeginRendering(command_buffer, rendering_info)

        # bind the graphics pipeline to the command buffer
        self._device.device.vkCmdBindPipeline(
            command_buffer,
            vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._pipeline,
        )

        # set the dynamic states for the pipeline
        self._device.device.vkCmdSetViewport(command_buffer, 0, 1, pointer(self._viewport))
        self._device.device.vkCmdSetScissor(command_buffer, 0, 1, pointer(self._scissor))

        # draw the triangle (3 vertices, 1 instance, first vertex = 0, first instance = 0)
        self._device.device.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        # finish the rendering commands
        self._device.device.vkCmdEndRendering(command_buffer)

        transition_image_layout(
            device=self._device.device,
            command_buffer=command_buffer,
            image=self._swapchain.swapchain_images[image_index],
            old_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            new_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            src_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            dst_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_NONE,
            src_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            dst_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        )

        # self._device.device.vkCmdEnd(command_buffer)

    def resize(self, width: int = 0, height: int = 0) -> None:
        """
        Resize the triangle pass.

        Args:
            width (int, optional): Width of the rendering area. If left at 0, the swapchain extent is used.
            height (int, optional): Height of the rendering area. If left at 0, the swapchain extent is used.

        """
        # define the viewport, which represents the area of the framebuffer to render to
        self._viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self._swapchain.swapchain_extent.width if width == 0 else width),
            height=float(self._swapchain.swapchain_extent.height if height == 0 else height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        # define the scissor, which represents a filter on the area of the framebuffer to render to
        self._scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=vk.VkExtent2D(
                width=self._swapchain.swapchain_extent.width if width == 0 else width,
                height=self._swapchain.swapchain_extent.height if height == 0 else height,
            ),
        )

    def close(self) -> None:
        """Release resources owned by this pass."""
        self._device.device.vkDestroyPipeline(self._pipeline, None)
        self._device.device.vkDestroyPipelineLayout(self._pipeline_layout, None)

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

        # define the variables in the vertex shader and how they are laid out in memory
        vertex_input = vk.VkPipelineVertexInputStateCreateInfo()
        # define the metadata of the pipeline,
        # for example the primitive to use to render the vertices
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        # define the viewport, which represents the area of the framebuffer to render to
        self._viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self._swapchain.swapchain_extent.width),
            height=float(self._swapchain.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        # define the scissor, which represents a filter on the area of the framebuffer to render to
        self._scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=self._swapchain.swapchain_extent,
        )

        # define the dynamic states for the pipeline
        # allowing the user to change these configurations at runtime
        # without recreating the pipeline
        dynamic_states = (c_int32 * 2)(
            vk.VkDynamicState.VK_DYNAMIC_STATE_VIEWPORT, vk.VkDynamicState.VK_DYNAMIC_STATE_SCISSOR
        )
        dynamic_state = vk.VkPipelineDynamicStateCreateInfo(
            dynamicStateCount=len(dynamic_states), pDynamicStates=dynamic_states
        )

        # create the viewport state (simple here cause of dynamic states)
        viewport_state = vk.VkPipelineViewportStateCreateInfo(viewportCount=1, scissorCount=1)

        # define the rasterizer, which converts the vertices into fragments
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            polygonMode=vk.VkPolygonMode.VK_POLYGON_MODE_FILL,  # define how the fragments are generated from the vertices
            depthClampEnable=vk.VK_FALSE,  # clamp the depth values instead of discarding fragments outside the near/far planes
            rasterizerDiscardEnable=vk.VK_FALSE,
            cullMode=vk.VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            lineWidth=1.0,
        )

        # configure the pipeline to enable hardware multisampling, which can help reduce aliasing artifacts
        multisample = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT
        )

        # define how colors should be blended together when rendering to the framebuffer
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=0,
            colorWriteMask=vk.VkColorComponentFlagBits.VK_COLOR_COMPONENT_R_BIT
            | vk.VkColorComponentFlagBits.VK_COLOR_COMPONENT_G_BIT
            | vk.VkColorComponentFlagBits.VK_COLOR_COMPONENT_B_BIT
            | vk.VkColorComponentFlagBits.VK_COLOR_COMPONENT_A_BIT,
        )
        color_blend = vk.VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1,
            pAttachments=pointer(color_blend_attachment),
        )

        # create the pipeline layout,
        # which defines the interface between the shader stages and the application
        # (uniforms)
        pipeline_layout = vk.VkPipelineLayout()
        check(
            self._device.device.vkCreatePipelineLayout(
                vk.VkPipelineLayoutCreateInfo(setLayoutCount=0, pushConstantRangeCount=0),
                None,
                byref(pipeline_layout),
            ),
            "vkCreatePipelineLayout",
        )
        self._pipeline_layout = pipeline_layout

        color_formats = (c_int32 * 1)(self._swapchain.swapchain_format)
        # create the metadata for the pipeline for dynamic rendering
        pipeline_rendering_info = vk.VkPipelineRenderingCreateInfo(
            colorAttachmentCount=len(color_formats), pColorAttachmentFormats=color_formats
        )

        # create the pipeline with all the provided info
        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2,
            pStages=stages,
            pVertexInputState=pointer(vertex_input),
            pInputAssemblyState=pointer(input_assembly),
            pViewportState=pointer(viewport_state),
            pRasterizationState=pointer(rasterizer),
            pMultisampleState=pointer(multisample),
            pColorBlendState=pointer(color_blend),
            pDynamicState=pointer(dynamic_state),
            layout=pipeline_layout,
            renderPass=vk.VkRenderPass(0),  # nullptr
            subpass=0,
            pNext=cast(
                pointer(pipeline_rendering_info), c_void_p
            ),  # link the dynamic rendering info to the pipeline
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
