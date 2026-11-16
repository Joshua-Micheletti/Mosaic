"""Vulkan Shader Utils Module."""

from ctypes import Array, byref, c_uint32

import vulkan_py as vk

from mosaic.engine.utils.vulkan.vulkan_utils.debug import check


def load_shader_module(device: vk.Device, path: str) -> vk.VkShaderModule:
    """
    Load a shader as a shader module.

    Args:
        device (vk.Device): Vulkan device.
        path (str): Path to the shader file.

    Returns:
        vk.VkShaderModule: The created shader module.

    """
    with open(path, "rb") as shader_file:
        code = shader_file.read()
    word_count = len(code) // 4
    buffer: Array[c_uint32] = (c_uint32 * word_count).from_buffer_copy(code)
    create_info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=buffer)
    module = vk.VkShaderModule()
    check(
        device.vkCreateShaderModule(create_info, None, byref(module)),
        "vkCreateShaderModule",
    )
    return module
