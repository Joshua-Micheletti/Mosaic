"""Vulkan pass that owns and draws the starter triangle."""

from __future__ import annotations
from vulkan_py.handles import VkDescriptorSet
from _ctypes import Array

import ctypes
import struct
import time
from ctypes import byref, c_int32, c_void_p, cast, pointer
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import vulkan_py as vk
from pyglm import glm

from mosaic.engine.renderer.vulkan_renderer.vulkan_render_stage import RenderStage
from mosaic.engine.utils.vulkan.debug import check
from mosaic.engine.utils.vulkan.memory import (
    Vertex,
    create_and_fill_gpu_buffer,
    create_buffer,
)
from mosaic.engine.utils.vulkan.shader import load_shader_module
from mosaic.engine.utils.vulkan.transition_image_layout import (
    ImageLayoutTransition,
    transition_image_layout,
)

if TYPE_CHECKING:
    from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
    from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain


@dataclass(slots=True)
class UniformBufferObject:
    model: glm.mat4
    view: glm.mat4
    proj: glm.mat4

    def to_bytes(self) -> bytes:
        values = (
            float(matrix[column][row])
            for matrix in (self.model, self.view, self.proj)
            for column in range(4)
            for row in range(4)
        )
        return struct.pack("<48f", *values)


class TrianglePass(RenderStage):
    """Create the triangle's Vulkan resources and record its draw."""

    _pipeline_layout: vk.VkPipelineLayout
    _pipeline: vk.VkPipeline
    _viewport: vk.VkViewport
    _scissor: vk.VkRect2D
    _command_pool: vk.VkCommandPool
    _uniform_buffers: list[vk.VkBuffer]
    _uniform_buffers_memory: list[vk.VkDeviceMemory]
    _uniform_buffers_mapped: list[Any]
    _start_time: float
    _descriptor_pool: vk.VkDescriptorPool
    _descriptor_set_layout: vk.VkDescriptorSetLayout

    def __init__(
        self,
        device: VulkanDevice,
        swapchain: VulkanSwapchain,
        command_pool: vk.VkCommandPool
    ) -> None:
        self._start_time = time.perf_counter()

        self._device = device
        self._swapchain = swapchain
        self._command_pool = command_pool
        self._create_descriptor_set_layout()
        self._create_graphics_pipeline()
        self._create_uniform_buffers()
        self._create_descriptor_pool()
        self._create_descriptor_sets()

    @override
    def record(
        self, command_buffer: vk.VkCommandBuffer, image_index: int
    ) -> None:
        # Before starting rendering, transition the swapchain image to ColorAttachmentOptimal
        transition_image_layout(
            device=self._device.device,
            command_buffer=command_buffer,
            image=self._swapchain.swapchain_images[image_index],
            transition=ImageLayoutTransition(
                old_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_NONE,
                dst_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                src_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                dst_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            ),
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

        vertex_buffers = (vk.VkBuffer * 1)(self._vertex_buffer)
        offsets = (ctypes.c_uint64 * 1)(0)

        self._device.device.vkCmdBindVertexBuffers(
            command_buffer,
            0,  # first binding
            1,  # number of buffers
            vertex_buffers,
            offsets,
        )

        self._device.device.vkCmdBindIndexBuffer(
            command_buffer, self._index_buffer, 0, vk.VkIndexType.VK_INDEX_TYPE_UINT32
        )

        # set the dynamic states for the pipeline
        self._device.device.vkCmdSetViewport(command_buffer, 0, 1, pointer(self._viewport))
        self._device.device.vkCmdSetScissor(command_buffer, 0, 1, pointer(self._scissor))

        descriptor_sets: Array[VkDescriptorSet] = (vk.VkDescriptorSet * 1)(
            self._descriptor_sets[image_index]
        )

        self._device.device.vkCmdBindDescriptorSets(
            command_buffer,
            vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._pipeline_layout,
            0,  # firstSet
            1,  # descriptorSetCount
            descriptor_sets,
            0,  # dynamicOffsetCount
            None,  # pDynamicOffsets
        )

        # draw the triangle (3 vertices, 1 instance, first vertex = 0, first instance = 0)
        # self._device.device.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self._device.device.vkCmdDrawIndexed(
            command_buffer,
            6,  # index count
            1,  # instance count
            0,  # first index
            0,  # vertex offset
            0,  # first instance
        )
        # finish the rendering commands
        self._device.device.vkCmdEndRendering(command_buffer)

        transition_image_layout(
            device=self._device.device,
            command_buffer=command_buffer,
            image=self._swapchain.swapchain_images[image_index],
            transition=ImageLayoutTransition(
                old_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                src_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask=vk.VkAccessFlagBits2.VK_ACCESS_2_NONE,
                src_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                dst_stage_mask=vk.VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            ),
        )

    @override
    def update(self, frame_index: int) -> None:
        elapsed_seconds = time.perf_counter() - self._start_time

        ubo = UniformBufferObject(
            model=glm.rotate(
                glm.mat4(1.0), elapsed_seconds * glm.radians(90.0), glm.vec3(0.0, 0.0, 1.0)
            ),
            view=glm.lookAt(
                glm.vec3(2.0, 2.0, 2.0),
                glm.vec3(0.0, 0.0, 0.0),
                glm.vec3(0.0, 0.0, 1.0),
            ),
            proj=glm.perspective(glm.radians(45.0), 1280 / 720, 0.1, 10.0),
        )

        packed_ubo = ubo.to_bytes()

        ctypes.memmove(self._uniform_buffers_mapped[frame_index], packed_ubo, len(packed_ubo))

    @override
    def resize(self, width: int = 0, height: int = 0) -> None:
        """
        Resize the triangle pass.

        Args:
            width (int, optional): Width of the rendering area.
                                   If left at 0, the swapchain extent is used.
            height (int, optional): Height of the rendering area.
                                    If left at 0, the swapchain extent is used.

        """
        # define the viewport, which represents the area of the framebuffer to render to
        # height is flipped to account for glm opengl convention
        self._viewport = vk.VkViewport(
            x=0.0,
            y=float(self._swapchain.swapchain_extent.height if height == 0 else height),
            width=float(self._swapchain.swapchain_extent.width if width == 0 else width),
            height=-float(self._swapchain.swapchain_extent.height if height == 0 else height),
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

    @override
    def close(self) -> None:
        """Release resources owned by this pass."""
        self._device.device.vkDestroyBuffer(self._vertex_buffer, None)
        self._device.device.vkFreeMemory(self._vertex_buffer_memory, None)
        self._device.device.vkDestroyBuffer(self._index_buffer, None)
        self._device.device.vkFreeMemory(self._index_buffer_memory, None)
        self._device.device.vkDestroyPipeline(self._pipeline, None)
        self._device.device.vkDestroyPipelineLayout(self._pipeline_layout, None)
        self._device.device.vkDestroyDescriptorPool(self._descriptor_pool, None)

    def _create_descriptor_set_layout(self) -> None:
        ubo_layout_binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
        )

        ubo_layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount=1, pBindings=pointer(ubo_layout_binding)
        )

        descriptor_set_layout = vk.VkDescriptorSetLayout()
        check(
            self._device.device.vkCreateDescriptorSetLayout(
                ubo_layout_info,
                None,
                byref(descriptor_set_layout),
            ),
            "vkCreateDescriptorSetLayout",
        )
        self._descriptor_set_layout = descriptor_set_layout

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
        vertices = [
            Vertex(glm.vec2(-0.5, -0.5), glm.vec3(1.0, 0.0, 0.0)),
            Vertex(glm.vec2(0.5, -0.5), glm.vec3(0.0, 1.0, 0.0)),
            Vertex(glm.vec2(0.5, 0.5), glm.vec3(0.0, 0.0, 1.0)),
            Vertex(glm.vec2(-0.5, 0.5), glm.vec3(1.0, 1.0, 1.0)),
        ]

        indices = [0, 1, 2, 2, 3, 0]

        # pack the vertices in a byte configuration that is useful by the GPU
        packed_vertices = b"".join(
            struct.pack("<5f", *vertex.pos, *vertex.color) for vertex in vertices
        )
        packed_indices = struct.pack(f"<{len(indices)}I", *indices)

        # create the vertex buffer that is optimized for the GPU and invisible by the CPU
        (self._vertex_buffer, self._vertex_buffer_memory) = create_and_fill_gpu_buffer(
            packed_vertices,
            self._device,
            vk.VkBufferUsageFlagBits.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            self._command_pool,
        )

        # create the index buffer that is optimized for the GPU and invisible by the CPU
        (self._index_buffer, self._index_buffer_memory) = create_and_fill_gpu_buffer(
            packed_indices,
            self._device,
            vk.VkBufferUsageFlagBits.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            self._command_pool,
        )

        binding = vk.VkVertexInputBindingDescription(
            binding=0,  # define the binding position
            stride=20,  # define the length of the single data ((2 + 3) * 4)
            # specify if the data should change per vertex or per instance
            inputRate=vk.VkVertexInputRate.VK_VERTEX_INPUT_RATE_VERTEX,
        )

        bindings = (vk.VkVertexInputBindingDescription * 1)(binding)

        attribute_pos = vk.VkVertexInputAttributeDescription(
            location=0, binding=0, format=vk.VkFormat.VK_FORMAT_R32G32_SFLOAT, offset=0
        )

        attribute_col = vk.VkVertexInputAttributeDescription(
            location=1, binding=0, format=vk.VkFormat.VK_FORMAT_R32G32B32_SFLOAT, offset=8
        )

        attributes = (vk.VkVertexInputAttributeDescription * 2)(attribute_pos, attribute_col)

        vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=len(bindings),
            pVertexBindingDescriptions=bindings,
            vertexAttributeDescriptionCount=len(attributes),
            pVertexAttributeDescriptions=attributes,
        )
        # define the metadata of the pipeline,
        # for example the primitive to use to render the vertices
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        # define the viewport, which represents the area of the framebuffer to render to
        # height is flipped to account for opengl convention
        self._viewport = vk.VkViewport(
            x=0.0,
            y=float(self._swapchain.swapchain_extent.height),
            width=float(self._swapchain.swapchain_extent.width),
            height=-float(self._swapchain.swapchain_extent.height),
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
            # define how the fragments are generated from the vertices
            polygonMode=vk.VkPolygonMode.VK_POLYGON_MODE_FILL,
            # clamp the depth values instead of discarding fragments outside the near/far planes
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            cullMode=vk.VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VkFrontFace.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            lineWidth=1.0,
        )

        # configure the pipeline to enable hardware multisampling,
        # which can help reduce aliasing artifacts
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
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=pointer(self._descriptor_set_layout),
            pushConstantRangeCount=0,
        )

        pipeline_layout = vk.VkPipelineLayout()
        check(
            self._device.device.vkCreatePipelineLayout(
                pipeline_layout_info,
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

    def _create_uniform_buffers(self) -> None:
        sample_ubo = UniformBufferObject(
            model=glm.mat4(1.0),
            view=glm.mat4(1.0),
            proj=glm.mat4(1.0),
        )

        size = len(sample_ubo.to_bytes())

        self._uniform_buffers = []
        self._uniform_buffers_memory = []
        self._uniform_buffers_mapped = []

        for _ in range(len(self._swapchain.swapchain_images)):
            (buffer, buffer_memory) = create_buffer(
                data_length=size,
                device=self._device,
                usage=vk.VkBufferUsageFlagBits.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                properties=(
                    vk.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | vk.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ),
            )

            # mapping the memory before hand to not have to do it each cycle
            mapped = ctypes.c_void_p()

            check(
                result=self._device.device.vkMapMemory(
                    buffer_memory,
                    0,
                    size,
                    0,
                    ctypes.byref(mapped),
                ),
                op="vkMapMemory",
            )

            self._uniform_buffers.append(buffer)
            self._uniform_buffers_memory.append(buffer_memory)
            self._uniform_buffers_mapped.append(mapped)

    def _create_descriptor_pool(self) -> None:
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=len(self._swapchain.swapchain_images),
        )

        pool_info = vk.VkDescriptorPoolCreateInfo(
            flags=vk.VkDescriptorPoolCreateFlagBits.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=len(self._swapchain.swapchain_images),
            poolSizeCount=1,
            pPoolSizes=pointer(pool_size),
        )

        self._descriptor_pool = vk.VkDescriptorPool()
        check(
            self._device.device.vkCreateDescriptorPool(
                pool_info,
                None,
                byref(self._descriptor_pool),
            ),
            "vkCreateDescriptorPool",
        )

    def _create_descriptor_sets(self) -> None:
        layouts: Array[vk.VkDescriptorSetLayout] = (
            vk.VkDescriptorSetLayout * len(self._swapchain.swapchain_images)
        )(*([self._descriptor_set_layout] * len(self._swapchain.swapchain_images)))

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self._descriptor_pool,
            descriptorSetCount=len(self._swapchain.swapchain_images),
            pSetLayouts=layouts,
        )

        descriptor_sets: Array[vk.VkDescriptorSet] = (
            vk.VkDescriptorSet * len(self._swapchain.swapchain_images)
        )()
        check(
            result=self._device.device.vkAllocateDescriptorSets(alloc_info, descriptor_sets),
            op="vkAllocateDescriptorSets",
        )

        self._descriptor_sets = list(descriptor_sets)

        for i in range(len(self._swapchain.swapchain_images)):
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=self._uniform_buffers[i],
                offset=0,
                range=len(
                    UniformBufferObject(glm.mat4(1.0), glm.mat4(1.0), glm.mat4(1.0)).to_bytes()
                ),
            )

            descriptor_write = vk.VkWriteDescriptorSet(
                dstSet=self._descriptor_sets[i],
                dstBinding=0,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=pointer(buffer_info),
            )

            self._device.device.vkUpdateDescriptorSets(
                1,
                pointer(descriptor_write),
                0,
                None,
            )
