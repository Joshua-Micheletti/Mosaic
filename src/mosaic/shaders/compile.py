"""Module for compiling shaders."""

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    """Compile all .slang shader files in the same directory as this script to SPIR-V."""
    # Check that the compiler is available
    compiler = shutil.which("slangc")
    if compiler is None:
        error = "slangc was not found on PATH"
        raise FileNotFoundError(error)

    # get all the shader paths to compile
    shader_directory = Path(__file__).resolve().parent
    shader_sources = sorted(shader_directory.glob("*.slang"))
    if not shader_sources:
        error = f"No .slang shader files found in {shader_directory}"
        raise FileNotFoundError(error)

    shaders: list[tuple[Path, Path, str]] = []
    for source in shader_sources:
        extension = source.stem.rsplit(".", maxsplit=1)[-1]
        # extract the shader stage from the file extension and determine the output path
        if extension == "vert":
            stage = "vertex"
        elif extension == "frag":
            stage = "fragment"
        else:
            error = (
                f"Cannot determine shader stage from {source.name};"
                "expected a .vert.slang or .frag.slang filename"
            )
            raise ValueError(error)
        shaders.append((source, source.with_suffix(".spv"), stage))

    # call slangc to compile each shader
    for source, output, stage in shaders:
        subprocess.run(  # noqa: S603
            [
                compiler,
                str(source),
                "-target",
                "spirv",
                "-entry",
                "main",
                "-profile",
                "spirv_1_4",
                "-stage",
                stage,
                "-o",
                str(output),
            ],
            check=True,
        )
        print(f"Compiled {source.name} -> {output.name}")


if __name__ == "__main__":
    main()
