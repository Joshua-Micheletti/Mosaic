import vulkan_py as vk


def check(result, op):
    if result != vk.VkResult.VK_SUCCESS:
        raise RuntimeError(f"Operation {op} failed execution | result : {result}")
