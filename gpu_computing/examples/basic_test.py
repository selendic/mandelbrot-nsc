"""
basic_test.py — Minimal PyOpenCL installation test for L10.

Run this BEFORE the lecture to confirm PyOpenCL is working:
    python basic_test.py

Expected output:
    Device: <your GPU or CPU device name>
    Kernel output: [1. 4. 9. 16.]
    All elements close? True

If you see an error, see the troubleshooting notes in the L09 slides
or install PoCL (CPU fallback):
    conda install -c conda-forge pyopencl pocl

Adapted from 2024 course template.py.
"""

import numpy as np
import pyopencl as cl

# ---------------------------------------------------------------------------
# 1. Create context and command queue
# ---------------------------------------------------------------------------
ctx = cl.create_some_context(interactive=False)  # picks first available
queue = cl.CommandQueue(ctx)

dev = ctx.devices[0]
print(f"Device: {dev.name}")
print(f"  Vendor:  {dev.vendor}")
print(f"  OpenCL:  {dev.version}")
print(f"  Compute units: {dev.max_compute_units}")
print()

# ---------------------------------------------------------------------------
# 2. Kernel: square each element of a float32 array
# ---------------------------------------------------------------------------
kernel_source = """
__kernel void square(__global float *a) {
    int i = get_global_id(0);
    a[i] = a[i] * a[i];
}
"""

prog = cl.Program(ctx, kernel_source).build()

# ---------------------------------------------------------------------------
# 3. Allocate host and device buffers
# ---------------------------------------------------------------------------
a_host = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
result = np.empty_like(a_host)

mf = cl.mem_flags
a_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_host)

# ---------------------------------------------------------------------------
# 4. Launch kernel
# ---------------------------------------------------------------------------
prog.square(queue, a_host.shape, None, a_dev)

# ---------------------------------------------------------------------------
# 5. Copy result back to host
# ---------------------------------------------------------------------------
cl.enqueue_copy(queue, result, a_dev)
queue.finish()

print("Kernel output:", result)  # Expected: [1. 4. 9. 16.]

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
expected = a_host ** 2
ok = np.allclose(result, expected)
print("All elements close?", ok)

if not ok:
    print("MISMATCH! Expected:", expected, "  Got:", result)
    raise SystemExit(1)

"""
Device: NVIDIA GeForce RTX 5050 Laptop GPU
  Vendor:  NVIDIA Corporation
  OpenCL:  OpenCL 3.0 CUDA
  Compute units: 20

Kernel output: [ 1.  4.  9. 16.]
All elements close? True
"""
