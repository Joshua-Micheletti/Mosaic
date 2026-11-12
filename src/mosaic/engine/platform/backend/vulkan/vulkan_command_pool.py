from ctypes import byref, pointer

import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.platform.backend.vulkan.vulkan_graphics_pipeline import VulkanGraphicsPipeline
from mosaic.engine.platform.backend.vulkan.vulkan_swapchain import VulkanSwapchain
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check

MAX_FRAMES_IN_FLIGHT = 2

class VulkanCommandPool(Closable):
    _command_pool: vk.VkCommandPool
    _command_buffers: list[vk.VkCommandBuffer]
    _image_available_semaphores: list[vk.VkSemaphore]
    _in_flight_fences: list[vk.VkFence]
    _render_finished_semaphores: list[vk.VkSemaphore]
    _current_frame: int
    _device: VulkanDevice

    @property
    def command_pool(self) -> vk.VkCommandPool:
        return self._command_pool

    @property
    def command_buffers(self) -> list[vk.VkCommandBuffer]:
        return self._command_buffers

    @property
    def image_available_semaphores(self) -> list[vk.VkSemaphore]:
        return self._image_available_semaphores

    @property
    def in_flight_fences(self) -> list[vk.VkFence]:
        return self._in_flight_fences

    @property
    def render_finished_semaphores(self) -> list[vk.VkSemaphore]:
        return self._render_finished_semaphores

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def __init__(
        self,
        device: VulkanDevice,
        swapchain: VulkanSwapchain,
        graphics_pipeline: VulkanGraphicsPipeline,
    ) -> None:
        self._device = device
        self._create_command_pool(device)
        self._create_command_buffers(device, swapchain, graphics_pipeline)
        self._create_sync_objects(device, swapchain)

    def advance_frame(self) -> None:
        """Advance the frame."""
        self._current_frame = (self._current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def close(self) -> None:
        """Close the command pool."""
        for s in self._image_available_semaphores:
            self._device.device.vkDestroySemaphore(s, None)
        for s in self._render_finished_semaphores:
            self._device.device.vkDestroySemaphore(s, None)
        for f in self._in_flight_fences:
            self._device.device.vkDestroyFence(f, None)
        self._device.device.vkDestroyCommandPool(self._command_pool, None)

    def _create_command_pool(self, device: VulkanDevice) -> None:
        create_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=device.graphics_family,
        )
        pool = vk.VkCommandPool()
        check(
            device.device.vkCreateCommandPool(create_info, None, byref(pool)), "vkCreateCommandPool"
        )
        self._command_pool = pool

    def _create_command_buffers(
        self,
        device: VulkanDevice,
        swapchain: VulkanSwapchain,
        graphics_pipeline: VulkanGraphicsPipeline,
    ) -> None:
        n = len(graphics_pipeline.framebuffers)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self._command_pool,
            level=vk.VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=n,
        )
        buffers = (vk.VkCommandBuffer * n)()
        check(
            device.device.vkAllocateCommandBuffers(alloc_info, buffers), "vkAllocateCommandBuffers"
        )
        self._command_buffers = list(buffers)

        clear_value = vk.VkClearValue(color=vk.VkClearColorValue(float32=(0.01, 0.01, 0.02, 1.0)))

        for i, cmd in enumerate(self._command_buffers):
            check(
                device.device.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo()),
                "vkBeginCommandBuffer",
            )

            render_pass_info = vk.VkRenderPassBeginInfo(
                renderPass=swapchain.render_pass,
                framebuffer=graphics_pipeline.framebuffers[i],
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
                cmd,
                vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphics_pipeline.pipeline,
            )
            device.device.vkCmdDraw(cmd, 3, 1, 0, 0)
            device.device.vkCmdEndRenderPass(cmd)

            check(device.device.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

    def _create_sync_objects(self, device: VulkanDevice, swapchain: VulkanSwapchain) -> None:
        self._image_available_semaphores, self._in_flight_fences = [], []
        sem_info = vk.VkSemaphoreCreateInfo()
        fence_info = vk.VkFenceCreateInfo(
            flags=vk.VkFenceCreateFlagBits.VK_FENCE_CREATE_SIGNALED_BIT
        )

        for _ in range(MAX_FRAMES_IN_FLIGHT):
            s1, f = vk.VkSemaphore(), vk.VkFence()
            check(device.device.vkCreateSemaphore(sem_info, None, byref(s1)), "vkCreateSemaphore")
            check(device.device.vkCreateFence(fence_info, None, byref(f)), "vkCreateFence")
            self._image_available_semaphores.append(s1)
            self._in_flight_fences.append(f)

        # one render-finished semaphore per swapchain image, not per frame-in-flight
        self._render_finished_semaphores = []
        for _ in range(len(swapchain.swapchain_images)):
            s2 = vk.VkSemaphore()
            check(device.device.vkCreateSemaphore(sem_info, None, byref(s2)), "vkCreateSemaphore")
            self._render_finished_semaphores.append(s2)

        self._current_frame = 0

