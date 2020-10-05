extern "C" __constant__ int my_constant = 314;

DEVICE FORCEINLINE f32 noise(V2f32 p) { return noise(p.x, p.y); }
DEVICE FORCEINLINE f32 noise(V3f32 p) { return noise(p.x, p.y, p.z); }

extern "C" __global__ void sum(const float* x, const float* y, float* out,
                               int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}