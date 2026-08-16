"""Metal-accelerated INT4 matvec, driven from Python via PyObjC.

No Xcode is required. `xcrun metal` (the offline shader compiler) needs a full
Xcode install and this machine only has the Command Line Tools -- but that
compiler is a separate thing from the Metal *framework*, which can compile MSL
source at runtime through `-newLibraryWithSource:options:error:`. That is the
same mechanism llama.cpp's Metal backend uses to ship .metal source as a string
rather than a prebuilt .metallib, and it is what this module does: the kernel
source in kernels/int4_matvec.metal is compiled once, on first use, by the
driver already sitting in every macOS install.

The design choice that matters: weights are uploaded to a GPU buffer ONCE, at
construction, and reused for every call. Only the activation vector `x`
(3.5 KB for hidden=896) is uploaded per token. Re-uploading a multi-megabyte
weight matrix every decode step would make the kernel slower than the CPU
path regardless of how fast the arithmetic is -- the KV cache lesson from
phase 4 (preallocate once, don't redo the expensive part every step) applies
here too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .quant import QuantConfig, QuantizedTensor, quantize

_KERNELS_DIR = Path(__file__).parent / "kernels"
_KERNEL_SRC = "\n".join(
    (_KERNELS_DIR / name).read_text()
    for name in ("int4_matvec.metal", "int4_matmat.metal")
)

_device = None
_queue = None
_pipeline = None       # int4_matvec: batch=1
_pipeline_batched = None  # int4_matmat: batch>1, one dispatch for the whole batch


def available() -> bool:
    """False on non-Apple hardware, or if PyObjC's Metal bindings are missing."""
    try:
        import Metal  # noqa: F401
        return Metal.MTLCreateSystemDefaultDevice() is not None
    except ImportError:
        return False


def _ensure_pipeline():
    global _device, _queue, _pipeline, _pipeline_batched
    if _pipeline is not None:
        return _device, _queue, _pipeline, _pipeline_batched

    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("no Metal device available")

    opts = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(_KERNEL_SRC, opts, None)
    if library is None:
        raise RuntimeError(f"Metal shader compile failed: {err}")

    def make_pipeline(fn_name: str):
        fn = library.newFunctionWithName_(fn_name)
        pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Metal pipeline creation failed for {fn_name}: {err}")
        return pipeline

    pipeline = make_pipeline("int4_matvec")
    pipeline_batched = make_pipeline("int4_matmat")

    _device, _queue = device, device.newCommandQueue()
    _pipeline, _pipeline_batched = pipeline, pipeline_batched
    return _device, _queue, _pipeline, _pipeline_batched


def _shared_buffer(device, arr: np.ndarray):
    import Metal
    arr = np.ascontiguousarray(arr)
    buf = device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
    )
    return buf


def _u32_buffer(device, value: int):
    import Metal
    b = np.uint32(value).tobytes()
    return device.newBufferWithBytes_length_options_(
        b, 4, Metal.MTLResourceStorageModeShared
    )


class MetalInt4Linear:
    """A single INT4-quantized linear layer, resident on the GPU.

    `weight` is (out_features, in_features), matching the layout every
    projection in tinyinfer/model.py already uses (`x @ weight.T`). Construction
    quantizes and uploads once; `__call__` costs one small upload (`x`) and one
    dispatch per call, which is the shape a decode step needs.
    """

    def __init__(self, weight: np.ndarray, config: QuantConfig | None = None):
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")
        config = config or QuantConfig(bits=4, group_size=128, symmetric=False)
        if config.bits != 4:
            raise ValueError("MetalInt4Linear only supports 4-bit codes "
                              "(the shader packs one byte per code, 0-15)")

        self.out_features, self.in_features = weight.shape
        self.qt: QuantizedTensor = quantize(weight, config)
        self.n_groups = self.qt.scale.shape[1]
        self.padded_cols = self.qt.q.shape[1] * self.qt.q.shape[2]

        device, self.queue, self.pipeline, self.pipeline_batched = _ensure_pipeline()
        self.device = device

        q_flat = self.qt.q.reshape(self.out_features, -1)
        self._q_buf = _shared_buffer(device, q_flat)
        self._s_buf = _shared_buffer(device, self.qt.scale)
        self._z_buf = _shared_buffer(device, self.qt.zero)
        self._in_f_buf = _u32_buffer(device, self.in_features)
        self._ng_buf = _u32_buffer(device, self.n_groups)
        self._gs_buf = _u32_buffer(device, config.group_size)
        self._pc_buf = _u32_buffer(device, self.padded_cols)

        import Metal
        self._y_buf = device.newBufferWithLength_options_(
            self.out_features * 4, Metal.MTLResourceStorageModeShared
        )
        self._tg = Metal.MTLSizeMake(
            min(self.out_features, self.pipeline.maxTotalThreadsPerThreadgroup()), 1, 1
        )
        self._grid = Metal.MTLSizeMake(self.out_features, 1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.in_features,):
            raise ValueError(f"expected x of shape ({self.in_features},), got {x.shape}")

        x_buf = _shared_buffer(self.device, x.astype(np.float32, copy=False))

        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)
        enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(self._q_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(self._s_buf, 0, 2)
        enc.setBuffer_offset_atIndex_(self._z_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(self._y_buf, 0, 4)
        enc.setBuffer_offset_atIndex_(self._in_f_buf, 0, 5)
        enc.setBuffer_offset_atIndex_(self._ng_buf, 0, 6)
        enc.setBuffer_offset_atIndex_(self._gs_buf, 0, 7)
        enc.setBuffer_offset_atIndex_(self._pc_buf, 0, 8)
        enc.dispatchThreads_threadsPerThreadgroup_(self._grid, self._tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        mv = self._y_buf.contents().as_buffer(self.out_features * 4)
        return np.frombuffer(mv, dtype=np.float32).copy()

    def batched(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, in_features) -> (batch, out_features), one dispatch total.

        This is the point of the batched kernel: `batch` calls to `__call__`
        would pay the ~0.24ms fixed dispatch cost `batch` times (see
        LOG.md, 2026-08-16, on the batch=1 result). This pays it once, for the
        whole batch, by using a 2D thread grid (out_features x batch) in a
        single command buffer.
        """
        import Metal

        if X.ndim != 2 or X.shape[1] != self.in_features:
            raise ValueError(f"expected X of shape (batch, {self.in_features}), "
                             f"got {X.shape}")
        batch = X.shape[0]

        x_buf = _shared_buffer(self.device, X.astype(np.float32, copy=False))
        out_f_buf = _u32_buffer(self.device, self.out_features)
        y_buf = self.device.newBufferWithLength_options_(
            batch * self.out_features * 4, Metal.MTLResourceStorageModeShared
        )

        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline_batched)
        enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(self._q_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(self._s_buf, 0, 2)
        enc.setBuffer_offset_atIndex_(self._z_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(y_buf, 0, 4)
        enc.setBuffer_offset_atIndex_(self._in_f_buf, 0, 5)
        enc.setBuffer_offset_atIndex_(self._ng_buf, 0, 6)
        enc.setBuffer_offset_atIndex_(self._gs_buf, 0, 7)
        enc.setBuffer_offset_atIndex_(self._pc_buf, 0, 8)
        enc.setBuffer_offset_atIndex_(out_f_buf, 0, 9)

        grid = Metal.MTLSizeMake(self.out_features, batch, 1)
        # Threadgroup width capped by the pipeline's own limit; height 1 keeps
        # each group within a single batch row so no group spans two rows.
        tg_w = min(self.out_features, self.pipeline_batched.maxTotalThreadsPerThreadgroup())
        tg = Metal.MTLSizeMake(tg_w, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        mv = y_buf.contents().as_buffer(batch * self.out_features * 4)
        return np.frombuffer(mv, dtype=np.float32).reshape(batch, self.out_features).copy()

    @property
    def compression(self) -> float:
        return self.qt.fp32_bytes / self.qt.stored_bytes
