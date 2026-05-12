

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
    NSString* metalPath = [dir stringByAppendingPathComponent:@"ddpm_step.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read ddpm_step.metal: ",
                [[error localizedDescription] UTF8String]);

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile ddpm_step.metal: ",
                [[error localizedDescription] UTF8String]);

    id<MTLFunction> func = [_library newFunctionWithName:@"ddpm_step_kernel"];
    TORCH_CHECK(func != nil, "ddpm_step_kernel function not found in metal source");

    _pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create pipeline state: ",
                [[error localizedDescription] UTF8String]);

    return _pipeline;
}

torch::Tensor ddpm_step_metal(
    torch::Tensor z_t,
    torch::Tensor eps,
    torch::Tensor noise,
    torch::Tensor alpha2,
    torch::Tensor alpha2_next,
    double var_lambda
) {
    TORCH_CHECK(z_t.is_mps(), "z_t must be on MPS device");
    TORCH_CHECK(z_t.dtype() == torch::kFloat32, "z_t must be float32");
    TORCH_CHECK(z_t.is_contiguous(), "z_t must be contiguous");
    TORCH_CHECK(eps.is_contiguous() && noise.is_contiguous(), "eps, noise must be contiguous");
    TORCH_CHECK(alpha2.is_contiguous() && alpha2_next.is_contiguous(), "alpha tensors must be contiguous");

    uint32_t B = z_t.size(0);
    uint32_t D = z_t.size(1);
    uint32_t total = B * D;
    float vl = (float)var_lambda;

    auto z_out = torch::empty_like(z_t);

    id<MTLComputePipelineState> pipeline = get_pipeline();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(z_t)         offset:z_t.storage_offset() * z_t.element_size()                  atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(eps)         offset:eps.storage_offset() * eps.element_size()                  atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(noise)       offset:noise.storage_offset() * noise.element_size()              atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2)      offset:alpha2.storage_offset() * alpha2.element_size()            atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2_next) offset:alpha2_next.storage_offset() * alpha2_next.element_size()  atIndex:4];

            [enc setBytes:&vl length:sizeof(float) atIndex:5];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(z_out)       offset:z_out.storage_offset() * z_out.element_size()              atIndex:6];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:7];

            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)total);
            MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return z_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ddpm_step", &ddpm_step_metal,
          "Fused DDPM denoising step (Metal compute shader)",
          py::arg("z_t"), py::arg("eps"), py::arg("noise"),
          py::arg("alpha2"), py::arg("alpha2_next"), py::arg("var_lambda"));
}
