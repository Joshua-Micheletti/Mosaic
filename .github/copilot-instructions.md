# Copilot Instructions for Mosaic

## Project overview

Mosaic is a Python Vulkan-based raster and path-tracing rendering engine. The project uses a src-layout package (`src/mosaic`) and is built with `uv`.

## Setup and validation commands

Install dependencies for the project and dev tooling:

```bash
uv sync --group dev
```

Run the application entry point:

```bash
uv run mosaic
# or
uv run python -m mosaic.main
```

Validated commands currently configured for this repo:

```bash
uv run ruff check src
uv run ty check src
```

There are no automated test files or pytest configuration checked in at the moment. Validation is currently driven by linting and static typing rather than a test suite.

## High-level architecture

The design is layered around the engine lifecycle and the platform/Vulkan backend:

- `src/mosaic/main.py` is the entry point and simply creates an `Application` and runs it.
- `src/mosaic/engine/core/application.py` owns the top-level application lifecycle and ensures the engine is closed during teardown.
- `src/mosaic/engine/core/engine.py` is the main runtime loop. It keeps the window alive until the user closes it, polls GLFW events, and asks the backend to render a frame.
- `src/mosaic/engine/platform/platform.py` is the platform façade: it initializes GLFW and creates the window and Vulkan backend.
- `src/mosaic/engine/platform/window.py` wraps the GLFW window and sets `glfw.CLIENT_API` to `glfw.NO_API`, which is important because this project relies on Vulkan instead of an OpenGL context.
- `src/mosaic/engine/platform/backend/vulkan/` contains the low-level Vulkan implementation. The backend is composed of these stages:
  - `VulkanSurface`: instance, validation layer/debug messenger, and presentation surface
  - `VulkanDevice`: physical-device selection and logical-device creation
  - `VulkanSwapchain`: swapchain and framebuffer setup
  - `VulkanGraphicsPipeline`: pipeline state for rendering
  - `VulkanCommandPool`: command buffers and frame synchronization
  - `VulkanBackend`: assembles the above pieces and drives `draw_frame()`
- `src/mosaic/engine/utils/lifecycle/` provides lifecycle abstractions (`Runnable` and `Closable`) used by the engine and platform classes.

## Key conventions and repository-specific patterns

- Use the package-qualified import style already used throughout the repo (`from mosaic.engine...`), not relative imports.
- Keep Vulkan and GLFW setup confined to `engine/platform` and `engine/platform/backend/vulkan`; do not spread raw graphics calls across unrelated modules.
- The codebase strongly follows ownership/lifecycle cleanup: each object that allocates resources implements `close()` and parent objects shut down their children in reverse order.
- Low-level Vulkan calls are typically wrapped in `check(...)` from `mosaic.engine.utils.vulkan.vulkan_utils` and use `ctypes`/`vk.*` structures directly. When touching this layer, follow the same pattern instead of introducing higher-level wrappers casually.
- The repo is a work in progress, and linting currently reports issues in the engine layer. Treat `uv run ruff check src` as a required repo check before finishing changes, even though current output may fail until those issues are addressed.
- The project is Python 3.14-oriented (`requires-python = ">=3.14"` in `pyproject.toml`), and the backend is very dependent on platform-specific Vulkan configuration and window lifecycle management.

## Notes for future Copilot sessions

- Prefer small, focused edits in the `engine/platform` and `engine/core` layers that match the existing architecture instead of introducing a second rendering abstraction.
- When changing engine startup or teardown, verify the initialization/cleanup flow still follows `Application -> Engine -> Platform -> Window/Backend` ownership.
- If new work adds a module, keep it under the appropriate `mosaic.engine.*` namespace and maintain the existing lifecycle and import conventions.
