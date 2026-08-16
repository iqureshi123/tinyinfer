// INT4 group-wise dequantize-and-matvec, fused into one pass.
//
// The point of this kernel is what it does NOT do: it never materializes a
// float32 weight matrix. tinyinfer/quant.py's dequantize() produces the full
// fp32 tensor and then a separate BLAS call multiplies it -- for decode
// (batch=1) that means reading the *quantized* bytes once to dequantize, then
// reading the *dequantized* fp32 bytes again for the matmul: two passes over
// memory when the operation is memory-bound in the first place. This kernel
// reads the INT4 codes exactly once and accumulates directly.
//
// One thread per output feature. Each thread walks its row of codes, and
// within each group folds the per-group scale/zero into a single multiply-add
// using the algebraic identity:
//
//   sum_i x[i] * (q[i]*scale + zero)
//     = scale * sum_i x[i]*q[i]  +  zero * sum_i x[i]
//
// which needs one accumulator for the weighted sum and one for the plain sum
// of x within the group, rather than dequantizing q[i] to a float before
// every multiply.
//
// Layout, matching tinyinfer.quant.QuantizedTensor exactly:
//   q      (out_features, n_groups, group_size) uint8, codes in [0, 15]
//   scale  (out_features, n_groups) float32
//   zero   (out_features, n_groups) float32
// q's last two dims are contiguous, so flattened per row it is (padded_cols,)
// with padding (if any) only past the real `in_features` -- the loop below
// never reads past in_features, so padding is never touched.

#include <metal_stdlib>
using namespace metal;

kernel void int4_matvec(
    device const float* x          [[buffer(0)]],  // (in_features,)
    device const uchar* q          [[buffer(1)]],  // (out_features, padded_cols)
    device const float* scale      [[buffer(2)]],  // (out_features, n_groups)
    device const float* zero       [[buffer(3)]],  // (out_features, n_groups)
    device float*       y          [[buffer(4)]],  // (out_features,)
    constant uint&       in_features [[buffer(5)]],
    constant uint&       n_groups    [[buffer(6)]],
    constant uint&       group_size  [[buffer(7)]],
    constant uint&       padded_cols [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    device const uchar* qrow = q + (size_t)gid * padded_cols;
    device const float* srow = scale + (size_t)gid * n_groups;
    device const float* zrow = zero  + (size_t)gid * n_groups;

    float acc = 0.0f;
    uint i = 0;
    for (uint g = 0; g < n_groups && i < in_features; g++) {
        uint end = min(i + group_size, in_features);
        float qsum = 0.0f;
        float xsum = 0.0f;
        for (uint j = i; j < end; j++) {
            float xj = x[j];
            qsum += xj * float(qrow[j]);
            xsum += xj;
        }
        acc += srow[g] * qsum + zrow[g] * xsum;
        i = end;
    }
    y[gid] = acc;
}
