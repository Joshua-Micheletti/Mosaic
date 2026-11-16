from typing import Protocol


class RenderDriver(Protocol):
    def run(self) -> None:
        pass

    def close(self) -> None:
        pass
