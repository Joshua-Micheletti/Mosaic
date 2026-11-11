"""Vulkan Surface."""

from ctypes import _Pointer, byref, c_char_p, c_void_p, pointer
from typing import TYPE_CHECKING

import glfw
import vulkan_py as vk

from mosaic.engine.utils.lifecycle.closable import Closable
from mosaic.engine.utils.vulkan.vulkan_utils import check

if TYPE_CHECKING:
    from ty_extensions._internal import Unknown


class VulkanSurface(Closable):
    """Vulkan Surface Class."""

    _instance_handle: vk.VkInstance
    _instance: vk.Instance
    _debug_callback_fn: Unknown
    _debug_messenger: vk.VkDebugUtilsMessengerEXT
    _surface: vk.VkSurfaceKHR

    @property
    def instance(self) -> vk.Instance:
        return self._instance

    @property
    def surface(self) -> vk.VkSurfaceKHR:
        return self._surface

    def __init__(self, window: _Pointer[glfw._GLFWwindow]) -> None:
        self._init_instance()
        self._init_debug()
        self._init_surface(window)

    def close(self) -> None:
        """Closing routine."""
        self._instance.vkDestroySurfaceKHR(self._surface, None)
        self._instance.vkDestroyDebugUtilsMessengerEXT(self._debug_messenger, None)
        self._instance.vkDestroyInstance(None)

    def _init_instance(self) -> None:
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

    def _init_debug(self) -> None:
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

    def _init_surface(self, window: _Pointer[glfw._GLFWwindow]) -> None:
        surface_handle = c_void_p(0)
        result = glfw.create_window_surface(
            self._instance_handle.value, window, None, byref(surface_handle)
        )
        if result != 0:
            error = f"glfwCreateWindowSurface failed: {result}"
            raise RuntimeError(error)

        if surface_handle.value is not None:
            self._surface = vk.VkSurfaceKHR(surface_handle.value)

    def _debug_callback(
        self,
        severity: int,
        _msg_type: int,
        callback_data_ptr: _Pointer[vk.VkDebugUtilsMessengerCallbackDataEXT],
        _user_data: c_void_p,
    ) -> int:
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
