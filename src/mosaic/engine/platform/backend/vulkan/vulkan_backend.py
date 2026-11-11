import ctypes
from ctypes import _Pointer, byref, c_uint32, pointer
from typing import TYPE_CHECKING

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_surface import VulkanSurface
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check

if TYPE_CHECKING:
    import glfw

MAX_FRAMES_IN_FLIGHT = 2
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class VulkanBackend(Closable):
    """Vulkan Backend class."""

    _device: VulkanDevice
    _surface: VulkanSurface

    @property
    def device(self) -> VulkanDevice:
        return self._device

    @property
    def surface(self) -> VulkanSurface:
        return self._surface

    def __init__(self, window: _Pointer[glfw._GLFWwindow]) -> None:
        self._surface = VulkanSurface(window)
        self._device = VulkanDevice(self._surface.instance, self._surface.surface)
        self._swapchain = VulkanSwapchain(window, self._device, self._surface)
        self._create_graphics_pipeline(self._device, self._swapchain)
        self._create_framebuffers(self._device, self._swapchain)
        self._create_command_pool(self._device)
        self._create_command_buffers(self._device, self._swapchain)
        self._create_sync_objects(self._device, self._swapchain)
        # https://github.com/StormMGamming41/vulkan-py/blob/main/examples/hello_triangle.py#L21

        # self._surface = glfw.create_window_surface()

    def close(self) -> None:
        """Closing routine."""
        for s in self.image_available_semaphores:
            self._device.device.vkDestroySemaphore(s, None)
        for s in self.render_finished_semaphores:
            self._device.device.vkDestroySemaphore(s, None)
        for f in self.in_flight_fences:
            self._device.device.vkDestroyFence(f, None)
        self._device.device.vkDestroyCommandPool(self.command_pool, None)
        for fb in self.framebuffers:
            self._device.device.vkDestroyFramebuffer(fb, None)
        self._device.device.vkDestroyPipeline(self.pipeline, None)
        self._device.device.vkDestroyPipelineLayout(self.pipeline_layout, None)
        self._device.device.vkDestroyRenderPass(self._swapchain.render_pass, None)
        for view in self._swapchain.swapchain_image_views:
            self._device.device.vkDestroyImageView(view, None)
        self._device.device.vkDestroySwapchainKHR(self._swapchain, None)
        self._device.device.vkDestroyDevice(None)
        self._surface.close()

    def _create_graphics_pipeline(self, device: VulkanDevice, swapchain: VulkanSwapchain):
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
        self.pipeline_layout = pipeline_layout

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
        self.pipeline = pipeline

        # shader modules aren't needed after pipeline creation
        device.device.vkDestroyShaderModule(vert_module, None)
        device.device.vkDestroyShaderModule(frag_module, None)

    def _create_framebuffers(self, device: VulkanDevice, swapchain: VulkanSwapchain):
        self.framebuffers = []
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
            self.framebuffers.append(fb)

    def _create_command_pool(self, device: VulkanDevice):
        create_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=device.graphics_family,
        )
        pool = vk.VkCommandPool()
        check(
            device.device.vkCreateCommandPool(create_info, None, byref(pool)), "vkCreateCommandPool"
        )
        self.command_pool = pool

    def _create_command_buffers(self, device: VulkanDevice, swapchain: VulkanSwapchain):
        n = len(self.framebuffers)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=n,
        )
        buffers = (vk.VkCommandBuffer * n)()
        check(
            device.device.vkAllocateCommandBuffers(alloc_info, buffers), "vkAllocateCommandBuffers"
        )
        self.command_buffers = list(buffers)

        clear_value = vk.VkClearValue(color=vk.VkClearColorValue(float32=(0.01, 0.01, 0.02, 1.0)))

        for i, cmd in enumerate(self.command_buffers):
            check(
                device.device.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo()),
                "vkBeginCommandBuffer",
            )

            render_pass_info = vk.VkRenderPassBeginInfo(
                renderPass=swapchain.render_pass,
                framebuffer=self.framebuffers[i],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0), extent=swapchain.swapchain_extent
                ),
                clearValueCount=1,
                pClearValues=pointer(clear_value),
            )
            device.device.vkCmdBeginRenderPass(
                cmd, byref(render_pass_info), vk.VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE
            )
            device.device.vkCmdBindPipeline(
                cmd, vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline
            )
            device.device.vkCmdDraw(cmd, 3, 1, 0, 0)
            device.device.vkCmdEndRenderPass(cmd)

            check(device.device.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

    def _create_sync_objects(self, device: VulkanDevice, swapchain: VulkanSwapchain):
        self.image_available_semaphores, self.in_flight_fences = [], []
        sem_info = vk.VkSemaphoreCreateInfo()
        fence_info = vk.VkFenceCreateInfo(
            flags=vk.VkFenceCreateFlagBits.VK_FENCE_CREATE_SIGNALED_BIT
        )

        for _ in range(MAX_FRAMES_IN_FLIGHT):
            s1, f = vk.VkSemaphore(), vk.VkFence()
            check(device.device.vkCreateSemaphore(sem_info, None, byref(s1)), "vkCreateSemaphore")
            check(device.device.vkCreateFence(fence_info, None, byref(f)), "vkCreateFence")
            self.image_available_semaphores.append(s1)
            self.in_flight_fences.append(f)

        # one render-finished semaphore per swapchain image, not per frame-in-flight
        self.render_finished_semaphores = []
        for _ in range(len(swapchain.swapchain_images)):
            s2 = vk.VkSemaphore()
            check(device.device.vkCreateSemaphore(sem_info, None, byref(s2)), "vkCreateSemaphore")
            self.render_finished_semaphores.append(s2)

        self.current_frame = 0

    def draw_frame(self):
        fence = self.in_flight_fences[self.current_frame]
        self._device.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)
        self._device.device.vkResetFences(1, byref(fence))

        image_index = c_uint32(0)
        self._device.device.vkAcquireNextImageKHR(
            self._swapchain.swapchain,
            UINT64_MAX,
            self.image_available_semaphores[self.current_frame],
            vk.VkFence(0),
            byref(image_index),
        )

        wait_semaphores = (vk.VkSemaphore * 1)(self.image_available_semaphores[self.current_frame])
        signal_semaphores = (vk.VkSemaphore * 1)(
            self.render_finished_semaphores[image_index.value]
        )  # <-- changed
        wait_stages = (ctypes.c_uint32 * 1)(
            vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        cmd_buffers = (vk.VkCommandBuffer * 1)(self.command_buffers[image_index.value])

        submit_info = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=1,
            pSignalSemaphores=signal_semaphores,
        )
        check(
            self._device.device.vkQueueSubmit(
                self._device.graphics_queue, 1, byref(submit_info), fence
            ),
            "vkQueueSubmit",
        )

        swapchains = (vk.VkSwapchainKHR * 1)(self._swapchain.swapchain)
        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=swapchains,
            pImageIndices=pointer(image_index),
        )
        self._device.device.vkQueuePresentKHR(self._device.present_queue, byref(present_info))

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def _load_shader_module(self, path, device: VulkanDevice):
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
