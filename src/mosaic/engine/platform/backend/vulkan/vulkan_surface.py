from mosaic.engine.utils.vulkan.vulkan_utils import check
from typing import cast
from ctypes import c_char_p, pointer, POINTER, byref

import glfw
import vulkan_py as vk

class VulkanSurface:

    def __init__(self) -> None:
       self._init_instance()
       self._init_debug()
       # https://github.com/StormMGamming41/vulkan-py/blob/main/examples/hello_triangle.py#L21

        #self._surface = glfw.create_window_surface()

    def _init_instance(self):
         # get the extensions required by GLFW
        extensions = glfw.get_required_instance_extensions()

        # create the application info
        app_info = vk.VkApplicationInfo(
            pApplicationName   = 'app'.encode('utf-8'), # We need to pass strings in bytes form
            applicationVersion = 1,
            pEngineName        = b"No Engine", # Same here
            engineVersion      = 1,
            apiVersion         = (1 << 22) # This tells Vulkan that we'll use API version 1.0
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
            pApplicationInfo        = pointer(app_info),
            enabledExtensionCount   = len(exts_bytes),
            ppEnabledExtensionNames = ext_array,
            enabledLayerCount       = len(layer_bytes),
            ppEnabledLayerNames     = layer_arr,
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

    def _debug_callback(self, severity, msg_type, callback_data_ptr, user_data):
        data = callback_data_ptr.contents
        message = data.pMessage.decode(errors="ignore") if data.pMessage else "<no message>"

        if severity >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            tag = "ERROR"
        elif severity >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            tag = "WARN"
        else:
            tag = "INFO"

        print(f"[validation:{tag}] {message}")
        return 0   # VK_FALSE - don't abort the call that triggered this