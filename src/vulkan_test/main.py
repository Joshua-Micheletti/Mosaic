"""Main Module."""

import glfw


def main() -> None:
    """
    Application entrypoint.

    Raises:
        RuntimeError: In case GLFW couldn't initialize
        RuntimeError: In case GLFW couldn't create a window

    """
    if not glfw.init():
        error = "Failed to initialize GLFW"
        raise RuntimeError(error)

    # We will use Vulkan, not an OpenGL context.
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    window = glfw.create_window(
        1280,
        720,
        "Vulkan Ray Tracer",
        None,
        None,
    )

    if not window:
        glfw.terminate()
        error = "Failed to create GLFW window"
        raise RuntimeError(error)

    try:
        print("GLFW window created successfully.")

        while not glfw.window_should_close(window):
            glfw.poll_events()

    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
