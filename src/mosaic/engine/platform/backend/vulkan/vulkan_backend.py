import ctypes
from ctypes import byref, c_char_p, c_uint32, c_void_p, pointer

import glfw
import vulkan_py as vk

from mosaic.engine.platform.backend.vulkan.vulkan_device import VulkanDevice
from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check

MAX_FRAMES_IN_FLIGHT = 2
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class VulkanBackend(Closable):
    def __init__(self, window) -> None:
        self._init_instance()
        self._init_debug()
        self._init_surface(window)
        self._device = VulkanDevice(self._instance, self._surface)
        self._create_swapchain(window, self._device)
        self._create_image_views(self._device)
        self._create_render_pass(self._device)
        self._create_graphics_pipeline(self._device)
        self._create_framebuffers(self._device)
        self._create_command_pool(self._device)
        self._create_command_buffers(self._device)
        self._create_sync_objects(self._device)
        # https://github.com/StormMGamming41/vulkan-py/blob/main/examples/hello_triangle.py#L21

        # self._surface = glfw.create_window_surface()

    def close(self) -> None:
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
        self._device.device.vkDestroyRenderPass(self.render_pass, None)
        for view in self.swapchain_image_views:
            self._device.device.vkDestroyImageView(view, None)
        self._device.device.vkDestroySwapchainKHR(self.swapchain, None)
        self._device.device.vkDestroyDevice(None)
        self._instance.vkDestroySurfaceKHR(self._surface, None)
        self._instance.vkDestroyDebugUtilsMessengerEXT(self._debug_messenger, None)
        self._instance.vkDestroyInstance(None)

    def _init_instance(self):
        # get the extensions required by GLFW
        extensions = glfw.get_required_instance_extensions()

        # create the application info
        app_info = vk.VkApplicationInfo(
            pApplicationName=b"app",  # We need to pass strings in bytes form
            applicationVersion=1,
            pEngineName=b"No Engine",  # Same here
            engineVersion=1,
            apiVersion=(1 << 22),  # This tells Vulkan that we'll use API version 1.0
        )

        # format the extension list in bytes and add the debug utils
        exts_bytes = [ext.encode("utf-8") for ext in extensions]
        exts_bytes.append(b"VK_EXT_debug_utils")
        ext_array = (c_char_p * len(exts_bytes))(*exts_bytes)

        # create the list of layers in bytes
        layer_bytes = [b"VK_LAYER_KHRONOS_validation"]
        layer_arr = (c_char_p * len(layer_bytes))(*layer_bytes)

        # create the info for the instance
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=pointer(app_info),
            enabledExtensionCount=len(exts_bytes),
            ppEnabledExtensionNames=ext_array,
            enabledLayerCount=len(layer_bytes),
            ppEnabledLayerNames=layer_arr,
        )

        # create the actual instance with the provided info
        self._instance_handle = vk.VkInstance()

        result = vk.vkCreateInstance(byref(create_info), None, byref(self._instance_handle))

        if result != vk.VkResult.VK_SUCCESS:
            # most likely VK_ERROR_LAYER_NOT_PRESENT - SDK's validation layer not installed
            print(f"[!] Instance creation with validation failed ({result}), retrying without it")
            create_info.enabledLayerCount = 0
            create_info.ppEnabledLayerNames = None
            create_info.enabledExtensionCount -= 1  # drop debug_utils too, nothing to feed it
            self._instance_handle = vk.VkInstance()
            result = vk.vkCreateInstance(byref(create_info), None, byref(self._instance_handle))

        check(result, "vkCreateInstance")
        self._instance = vk.Instance(self._instance_handle)

    def _init_debug(self):
        # Keep this CFUNCTYPE instance alive on self - if it gets garbage
        # collected, Vulkan is left holding a dangling function pointer and
        # will crash the next time a validation message fires.
        self._debug_callback_fn = vk.PFN_vkDebugUtilsMessengerCallbackEXT(self._debug_callback)

        create_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=(
                vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                | vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            ),
            messageType=(
                vk.VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | vk.VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | vk.VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            ),
            pfnUserCallback=self._debug_callback_fn,
        )

        messenger = vk.VkDebugUtilsMessengerEXT()
        result = self._instance.vkCreateDebugUtilsMessengerEXT(create_info, None, byref(messenger))
        check(result, "vkCreateDebugUtilsMessengerEXT")
        self._debug_messenger = messenger

    def _init_surface(self, window):
        surface_handle = c_void_p(0)
        result = glfw.create_window_surface(
            self._instance_handle.value, window, None, byref(surface_handle)
        )
        if result != 0:
            raise RuntimeError(f"glfwCreateWindowSurface failed: {result}")

        if surface_handle.value != None:
            self._surface = vk.VkSurfaceKHR(surface_handle.value)

    def _create_swapchain(self, window, device: VulkanDevice):
        caps = vk.VkSurfaceCapabilitiesKHR()
        self._instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device.physical_device, self._surface, byref(caps)
        )

        fmt_count = c_uint32(0)
        self._instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, self._surface, byref(fmt_count), None
        )
        formats = (vk.VkSurfaceFormatKHR * fmt_count.value)()
        self._instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device, self._surface, byref(fmt_count), formats
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
        self._instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, self._surface, byref(mode_count), None
        )
        modes = (ctypes.c_int32 * mode_count.value)()
        self._instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical_device, self._surface, byref(mode_count), modes
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
            surface=self._surface,
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
        self.swapchain, self.swapchain_format, self.swapchain_extent = (
            swapchain,
            chosen_format.format,
            extent,
        )

        img_count = c_uint32(0)
        device.device.vkGetSwapchainImagesKHR(self.swapchain, byref(img_count), None)
        images = (vk.VkImage * img_count.value)()
        device.device.vkGetSwapchainImagesKHR(self.swapchain, byref(img_count), images)
        self.swapchain_images = list(images)

    def _create_image_views(self, device: VulkanDevice):
        self.swapchain_image_views = []
        for img in self.swapchain_images:
            create_info = vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VkImageViewType.VK_IMAGE_VIEW_TYPE_2D,
                format=self.swapchain_format,
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
            self.swapchain_image_views.append(view)

    def _create_render_pass(self, device: VulkanDevice):
        color_attachment = vk.VkAttachmentDescription(
            format=self.swapchain_format,
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
        self.render_pass = render_pass

    def _create_graphics_pipeline(self, device: VulkanDevice):
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
            width=float(self.swapchain_extent.width),
            height=float(self.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        scissor = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=self.swapchain_extent)
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
            renderPass=self.render_pass,
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

    def _create_framebuffers(self, device: VulkanDevice):
        self.framebuffers = []
        for view in self.swapchain_image_views:
            create_info = vk.VkFramebufferCreateInfo(
                renderPass=self.render_pass,
                attachmentCount=1,
                pAttachments=pointer(view),
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
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

    def _create_command_buffers(self, device: VulkanDevice):
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
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[i],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0), extent=self.swapchain_extent
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

    def _create_sync_objects(self, device: VulkanDevice):
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
        for _ in range(len(self.swapchain_images)):
            s2 = vk.VkSemaphore()
            check(device.device.vkCreateSemaphore(sem_info, None, byref(s2)), "vkCreateSemaphore")
            self.render_finished_semaphores.append(s2)

        self.current_frame = 0

    def draw_frame(self, device: VulkanDevice):
        fence = self.in_flight_fences[self.current_frame]
        device.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)
        device.device.vkResetFences(1, byref(fence))

        image_index = c_uint32(0)
        device.device.vkAcquireNextImageKHR(
            self.swapchain,
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
            device.device.vkQueueSubmit(device.graphics_queue, 1, byref(submit_info), fence),
            "vkQueueSubmit",
        )

        swapchains = (vk.VkSwapchainKHR * 1)(self.swapchain)
        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=swapchains,
            pImageIndices=pointer(image_index),
        )
        device.device.vkQueuePresentKHR(device.present_queue, byref(present_info))

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def _debug_callback(self, severity, msg_type, callback_data_ptr, user_data):
        data = callback_data_ptr.contents
        message = data.pMessage.decode(errors="ignore") if data.pMessage else "<no message>"

        if (
            severity
            >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        ):
            tag = "ERROR"
        elif (
            severity
            >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        ):
            tag = "WARN"
        else:
            tag = "INFO"

        print(f"[validation:{tag}] {message}")
        return 0  # VK_FALSE - don't abort the call that triggered this

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
