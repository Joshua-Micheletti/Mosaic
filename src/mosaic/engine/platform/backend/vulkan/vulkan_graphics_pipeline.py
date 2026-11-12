from ctypes import Array, byref, c_uint32, pointer

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check


class VulkanGraphicsPipeline(Closable):
    _pipeline_layout: vk.VkPipelineLayout
    _pipeline: vk.VkPipeline
    _framebuffers: list[vk.VkFramebuffer]
    _device: VulkanDevice

    @property
    def pipeline_layout(self) -> vk.VkPipelineLayout:
        return self._pipeline_layout

    @property
    def pipeline(self) -> vk.VkPipeline:
        return self._pipeline

    @property
    def framebuffers(self) -> list[vk.VkFramebuffer]:
        return self._framebuffers

    def __init__(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        self._device = device
        self._create_graphics_pipeline(device, swapchain)
        self._create_framebuffers(device, swapchain)

    def close(self) -> None:
        """Close the graphics pipeline."""
        for fb in self._framebuffers:
            self._device.device.vkDestroyFramebuffer(fb, None)

        self._device.device.vkDestroyPipeline(self._pipeline, None)
        self._device.device.vkDestroyPipelineLayout(self._pipeline_layout, None)

    def _create_graphics_pipeline(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        vert_module, vert_buf = self._load_shader_module(
            "./src/mosaic/shaders/triangle.vert.spv", device
        )
        frag_module, frag_buf = self._load_shader_module(
            "./src/mosaic/shaders/triangle.frag.spv", device
        )

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
        stages = (vk.VkPipelineShaderStageCreateInfo * 2)(vert_stage, frag_stage)

        vertex_input = (
            vk.VkPipelineVertexInputStateCreateInfo()
        )  # no bindings - positions are hardcoded
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )

        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(swapchain.swapchain_extent.width),
            height=float(swapchain.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        scissor = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=swapchain.swapchain_extent)
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
            blendEnable=0, colorWriteMask=0xF
        )  # R|G|B|A
        color_blend = vk.VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1, pAttachments=pointer(color_blend_attachment)
        )

        pipeline_layout = vk.VkPipelineLayout()
        check(
            device.device.vkCreatePipelineLayout(
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
            renderPass=swapchain.render_pass,
            subpass=0,
        )
        pipeline = vk.VkPipeline()
        check(
            device.device.vkCreateGraphicsPipelines(
                vk.VkPipelineCache(0), 1, byref(pipeline_info), None, byref(pipeline)
            ),
            "vkCreateGraphicsPipelines",
        )
        self._pipeline = pipeline

        # shader modules aren't needed after pipeline creation
        device.device.vkDestroyShaderModule(vert_module, None)
        device.device.vkDestroyShaderModule(frag_module, None)

    def _create_framebuffers(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        self._framebuffers = []
        for view in swapchain.swapchain_image_views:
            create_info = vk.VkFramebufferCreateInfo(
                renderPass=swapchain.render_pass,
                attachmentCount=1,
                pAttachments=pointer(view),
                width=swapchain.swapchain_extent.width,
                height=swapchain.swapchain_extent.height,
                layers=1,
            )
            fb = vk.VkFramebuffer()
            check(
                device.device.vkCreateFramebuffer(create_info, None, byref(fb)),
                "vkCreateFramebuffer",
            )
            self._framebuffers.append(fb)

    def _load_shader_module(
        self, path: str, device: VulkanDevice
    ) -> tuple[vk.VkShaderModule, Array[c_uint32]]:
        with open(path, "rb") as f:
            code = f.read()
        word_count = len(code) // 4
        buf = (c_uint32 * word_count).from_buffer_copy(code)  # SPIR-V is a uint32 word stream
        create_info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=buf)
        module = vk.VkShaderModule()
        check(
            device.device.vkCreateShaderModule(create_info, None, byref(module)),
            "vkCreateShaderModule",
        )
        return module, buf  # keep buf alive - see warning below
