"""Runtime for kernels generated offline by FROST.

FROST is a build-time dependency only.  The deployed FlashInfer process reads
a sealed manifest and reloads the exported CuTeDSL object through TVM-FFI.
"""

__all__ = [
    "FrostGroupedGemm1SwiGLURunner",
    "matching_kernels",
    "workspace_size",
]


def __getattr__(name):
    # A deferred support-check import must not load the execution implementation.
    if name in __all__:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
