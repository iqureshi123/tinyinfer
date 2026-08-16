// Batched sibling of int4_matvec.metal.
//
// The batch=1 benchmark (see LOG.md, 2026-08-16) found that a Metal dispatch
// costs ~0.24ms of fixed command-buffer overhead almost regardless of the work
// inside it -- an 8x64 matrix and a 896x896 matrix dispatched about the same.
// At batch=1 that fixed cost dominates and buries the kernel's real advantage
// (reading 4x fewer weight bytes than fp32). The fix is not a faster kernel;
// it is fewer dispatches doing more work each. This kernel processes an entire
// batch of rows in ONE dispatch -- one command buffer, one submit, one wait --
// so the ~0.24ms is paid once for the whole batch instead of once per row.
//
// Same dequantization identity as int4_matvec.metal, just indexed by two
// dimensions instead of one: thread (o, b) computes output row b, feature o.

#include <metal_stdlib>
using namespace metal;

kernel void int4_matmat(
    device const float* x          [[buffer(0)]],  // (batch, in_features) row-major
    device const uchar* q          [[buffer(1)]],  // (out_features, padded_cols)
    device const float* scale      [[buffer(2)]],  // (out_features, n_groups)
    device const float* zero       [[buffer(3)]],  // (out_features, n_groups)
    device float*       y          [[buffer(4)]],  // (batch, out_features) row-major
    constant uint&       in_features  [[buffer(5)]],
    constant uint&       n_groups     [[buffer(6)]],
    constant uint&       group_size   [[buffer(7)]],
    constant uint&       padded_cols  [[buffer(8)]],
    constant uint&       out_features [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint o = gid.x;   // output feature
    uint b = gid.y;   // batch row

    device const float* xrow = x + (size_t)b * in_features;
    device const uchar* qrow = q + (size_t)o * padded_cols;
    device const float* srow = scale + (size_t)o * n_groups;
    device const float* zrow = zero  + (size_t)o * n_groups;

    float acc = 0.0f;
    uint i = 0;
    for (uint g = 0; g < n_groups && i < in_features; g++) {
        uint end = min(i + group_size, in_features);
        float qsum = 0.0f;
        float xsum = 0.0f;
        for (uint j = i; j < end; j++) {
            float xj = xrow[j];
            qsum += xj * float(qrow[j]);
            xsum += xj;
        }
        acc += srow[g] * qsum + zrow[g] * xsum;
        i = end;
    }
    y[(size_t)b * out_features + o] = acc;
}
