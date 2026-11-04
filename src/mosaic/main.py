"""Main Module."""

from mosaic.engine.core.application import Application


def main() -> None:
    """
    Application entrypoint.

    Raises:
        RuntimeError: In case GLFW couldn't initialize
        RuntimeError: In case GLFW couldn't create a window

    """
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
