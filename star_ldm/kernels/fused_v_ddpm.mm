

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLComputePipelineState> _pipeline = nil;
static id<MTLLibrary> _library = nil;

static id<MTLComputePipelineState> get_pipeline() {
    if (_pipeline != nil) return _pipeline;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    NSString* path = [NSString stringWithUTF8String:__FILE__];
    NSString* dir = [path stringByDeletingLastPathComponent];
    NSString* metalPath = [dir stringByAppendingPathComponent:@"fused_v_ddpm.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read fused_v_ddpm.metal: ",
                [[error localizedDescription] UTF8String]);

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile fused_v_ddpm.metal: ",
                [[error localizedDescription] UTF8String]);

    id<MTLFunction> func = [_library newFunctionWithName:@"fused_v_ddpm_kernel"];
    TORCH_CHECK(func != nil, "fused_v_ddpm_kernel function not found in metal source");

    _pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create pipeline state: ",
                [[error localizedDescription] UTF8String]);

    return _pipeline;
}

torch::Tensor fused_v_ddpm_metal(
    torch::Tensor z_t,
    torch::Tensor v_pred,
    torch::Tensor noise,
    torch::Tensor alpha2,
    torch::Tensor alpha2_next,
    double var_lambda,
    bool is_last_step
) {
    TORCH_CHECK(z_t.is_mps(), "z_t must be on MPS device");
    TORCH_CHECK(z_t.dtype() == torch::kFloat32, "z_t must be float32");
    TORCH_CHECK(z_t.is_contiguous(), "z_t must be contiguous");
    TORCH_CHECK(v_pred.is_contiguous() && noise.is_contiguous(), "v_pred, noise must be contiguous");
    TORCH_CHECK(alpha2.is_contiguous() && alpha2_next.is_contiguous(), "alpha tensors must be contiguous");

    uint32_t B = z_t.size(0);
    uint32_t D = z_t.size(1);
    uint32_t total = B * D;
    uint32_t is_last = is_last_step ? 1u : 0u;
    float vl = (float)var_lambda;

    auto out = torch::empty_like(z_t);

    id<MTLComputePipelineState> pipeline = get_pipeline();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(z_t)         offset:z_t.storage_offset() * z_t.element_size()                  atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(v_pred)      offset:v_pred.storage_offset() * v_pred.element_size()            atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(noise)       offset:noise.storage_offset() * noise.element_size()              atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2)      offset:alpha2.storage_offset() * alpha2.element_size()            atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2_next) offset:alpha2_next.storage_offset() * alpha2_next.element_size()  atIndex:4];

            [enc setBytes:&vl length:sizeof(float) atIndex:5];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)         offset:out.storage_offset() * out.element_size()                  atIndex:6];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&is_last length:sizeof(uint32_t) atIndex:8];

            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)total);
            MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_v_ddpm", &fused_v_ddpm_metal,
          "Fused v-prediction + DDPM denoising step (Metal compute shader)",
          py::arg("z_t"), py::arg("v_pred"), py::arg("noise"),
          py::arg("alpha2"), py::arg("alpha2_next"),
          py::arg("var_lambda"), py::arg("is_last_step"));
}
