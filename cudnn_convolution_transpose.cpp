#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:

    - cudnn_convolution
    - cudnn_convolution_backward_weight
    - cudnn_convolution_backward_input

The functions defined here can be called from Python in replacement of
torch.nn.conv2d, torch.nn.grad.conv2d_weight and torch.nn.grad.conv2d_input,
and run significantly faster. See 'example.py' for how these functions
are called.

Adapted from code posted by hanspinckaers:
https://discuss.pytorch.org/t/cuda-error-with-cudnn-convolution-backward-weight-function/41214
*/
at::Tensor convolution_transpose(
    const at::Tensor & input,
    const at::Tensor & weight,
    const c10::optional<at::Tensor> & bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic
){
    return at::cudnn_convolution_transpose(
        input,
        weight,
        bias,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic);
}

std::tuple<at::Tensor,at::Tensor> convolution_transpose_backward(
    const at::Tensor & input,
    const at::Tensor & grad_output,
    const at::Tensor & weight,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    std::array<bool,2> output_mask
)
{
    return at::cudnn_convolution_transpose_backward(
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
        output_mask);
}



at::Tensor convolution_transpose_backward_input(
    const at::Tensor & grad_output,
    const at::Tensor & weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32
){
    // return at::cudnn_convolution_backward_weight(
    return at::cudnn_convolution_transpose_backward_input(
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}


at::Tensor convolution_transpose_backward_weight(
    at::IntArrayRef weight_size,
    const at::Tensor & grad_output,
    const at::Tensor & weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32)
{
    return at::cudnn_convolution_transpose_backward_weight(
        weight_size,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convolution_transpose", &convolution_transpose, "convolution transpose");
    m.def("convolution_transpose_backward", &convolution_transpose_backward, "convolution transpose backward");
    m.def("convolution_transpose_backward_weight", &convolution_transpose_backward_weight, "convolution transpose backward weight");
    m.def("convolution_transpose_backward_input", &convolution_transpose_backward_input, "convolution transpose backward input");
}