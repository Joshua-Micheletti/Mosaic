from mosaic.engine.utils.lifecycle.closable import Closable
from ctypes import byref, c_char_p, c_float, c_uint32, pointer

import vulkan_py as vk

from mosaic.engine.utils.vulkan.vulkan_utils import check


class VulkanDevice(Closable):
    _physical_device: vk.VkPhysicalDevice
    _raw_device: vk.VkDevice
    _device: vk.Device

    _graphics_family: int
    _graphics_queue: vk.VkQueue

    _present_family: int
    _present_queue: vk.VkQueue

    @property
    def physical_device(self) -> vk.VkPhysicalDevice:
        return self._physical_device

    @property
    def raw_device(self) -> vk.VkDevice:
        return self._raw_device

    @property
    def device(self) -> vk.Device:
        return self._device

    @property
    def graphics_family(self) -> int:
        return self._graphics_family

    @property
    def graphics_queue(self) -> vk.VkQueue:
        return self._graphics_queue

    @property
    def present_family(self) -> int:
        return self._present_family

    @property
    def present_queue(self) -> vk.VkQueue:
        return self._present_queue

    def __init__(self, instance: vk.Instance, surface: vk.VkSurfaceKHR) -> None:
        self._pick_physical_device(instance, surface)
        self._create_logical_device(instance)

    def close(self) -> None:
        """Close the device."""
        self._device.device.vkDestroyDevice(None)

    def _pick_physical_device(self, instance: vk.Instance, surface: vk.VkSurfaceKHR) -> None:
        count = c_uint32(0)
        instance.vkEnumeratePhysicalDevices(byref(count), None)
        if count.value == 0:
            error = "No Vulkan-capable GPU found"
            raise RuntimeError(error)
        devices = (vk.VkPhysicalDevice * count.value)()
        instance.vkEnumeratePhysicalDevices(byref(count), devices)

        best = None
        for pd in devices:
            gfx, present = self._find_queue_families(pd, instance, surface)
            if gfx is None or present is None:
                continue
            props = vk.VkPhysicalDeviceProperties()
            instance.vkGetPhysicalDeviceProperties(pd, props)
            score = (
                2
                if props.deviceType == vk.VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                else 1
            )
            if best is None or score > best[0]:
                best = (score, pd, gfx, present, props.deviceName.decode(errors="ignore"))

        if best is None:
            error = "No GPU with both graphics and present support found"
            raise RuntimeError(error)

        _, pd, gfx, present, name = best
        self._physical_device, self._graphics_family, self._present_family = pd, gfx, present
        print(f"Using GPU: {name} (graphics family={gfx}, present family={present})")

    def _create_logical_device(self, instance: vk.Instance) -> None:
        unique_families = {self._graphics_family, self._present_family}
        priority = c_float(1.0)
        queue_infos = [
            vk.VkDeviceQueueCreateInfo(
                queueFamilyIndex=fam, queueCount=1, pQueuePriorities=pointer(priority)
            )
            for fam in unique_families
        ]
        queue_infos_arr = (vk.VkDeviceQueueCreateInfo * len(queue_infos))(*queue_infos)

        device_extensions = [b"VK_KHR_swapchain"]
        ext_arr = (c_char_p * len(device_extensions))(*device_extensions)
        features = vk.VkPhysicalDeviceFeatures()  # nothing extra needed yet

        create_info = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=queue_infos_arr,
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=ext_arr,
            pEnabledFeatures=pointer(features),
        )

        raw_device = vk.VkDevice()
        check(
            instance.vkCreateDevice(self._physical_device, create_info, None, raw_device),
            "vkCreateDevice",
        )
        self._raw_device = raw_device
        self._device = vk.Device(raw_device)

        graphics_queue = vk.VkQueue()
        self._device.vkGetDeviceQueue(self._graphics_family, 0, byref(graphics_queue))
        self._graphics_queue = graphics_queue

        present_queue = vk.VkQueue()
        self._device.vkGetDeviceQueue(self._present_family, 0, byref(present_queue))
        self._present_queue = present_queue

    def _find_queue_families(
        self, physical_device: vk.VkPhysicalDevice, instance: vk.Instance, surface: vk.VkSurfaceKHR
    ) -> tuple[int | None, int | None]:
        count = c_uint32(0)
        instance.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, byref(count), None)
        families = (vk.VkQueueFamilyProperties * count.value)()
        instance.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, byref(count), families)

        graphics_family = present_family = None
        for i, fam in enumerate(families):
            if fam.queueFlags & vk.VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT:
                graphics_family = i
            supported = c_uint32(0)
            instance.vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device, i, surface, byref(supported)
            )
            if supported.value:
                present_family = i
            if graphics_family is not None and present_family is not None:
                break

        return graphics_family, present_family
