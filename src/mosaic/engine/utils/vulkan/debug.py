"""Vulkan Debug Utils Module."""

from typing import TYPE_CHECKING

import vulkan_py as vk

if TYPE_CHECKING:
    from ty_extensions._internal import Unknown


def check(result: Unknown, op: str) -> None:
    """
    Check the result of a Vulkan operation and raise an exception if it failed.

    Args:
        result (Unknown): Return value of the executed vulkan operation.
        op (str): Name of the vulkan operation that was executed.

    Raises:
        RuntimeError: In case the vulkan operation failed.

    """
    if result != vk.VkResult.VK_SUCCESS:
        error = f"Operation {op} failed execution | result : {result}"
        raise RuntimeError(error)
