# from torch.nn import Function
from torch.autograd import Function
from torch import nn
from torch.nn import Linear
from torch import Tensor

from torch.utils.cpp_extension import load

# from torch import conv_transpose2d
cudnn_transpose_convolution = load(
    name="cudnn_convolution_transpose",
    sources=["cudnn_convolution_transpose.cpp"],
    verbose=True,
)


# Inherit from Function
class _custom_conv_transpose2d(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor = None,
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        groups=1,
        dilation=(1, 1),
    ):
        # ctx.save_for_backward(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.output_padding = output_padding

        # output=input.mm(weight.t())
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        output = cudnn_transpose_convolution.convolution_transpose(
            input,
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            False,  # benchmark,
            False,  # deterministic
        )
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        dilation = ctx.dilation
        groups = ctx.groups
        output_padding = ctx.output_padding
        grad_input: Tensor = None
        grad_weight: Tensor = None
        grad_bias: Tensor = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # NOTE: This doesn't currently work, this `convolution_transpose_backward`
        # returns None, None, rather than the tuple of tensors it is supposed to return! 
        # if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
        #     # Use a (perhaps) more efficient implementation that gets the grad of both?
        #     # grad_input=grad_output.mm(weight)
        #     grads =  cudnn_transpose_convolution.convolution_transpose_backward(
        #         input,
        #         grad_output,
        #         weight,
        #         padding,
        #         output_padding,
        #         stride,
        #         dilation,
        #         groups,
        #         False,  # benchmark
        #         False,  # deterministic
        #         False,  # allow_tf32
        #         [False, False],  # output_mask
        #     )
        #     assert False, grads
        #     grad_input, grad_weight = grads
        #     # assert False, (grad_input.shape, input.shape)

        # else:
        if ctx.needs_input_grad[0]:
            # grad_input=grad_output.mm(weight)
            grad_input = cudnn_transpose_convolution.convolution_transpose_backward_input(
                grad_output,
                weight,
                padding,
                stride,
                dilation,
                groups,
                False,
                False,
                False,
            )
        if ctx.needs_input_grad[1]:
            # grad_weight=grad_output.t().mm(input)
            grad_weight = cudnn_transpose_convolution.convolution_transpose_backward_weight(
                weight.shape,
                grad_output,
                input,
                padding,
                stride,
                dilation,
                groups,
                False,
                False,
                False,
            )
        if bias is not None and ctx.needs_input_grad[2] and grad_output is not None:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


custom_conv_transpose2d = _custom_conv_transpose2d.apply


# unrelated stuff, was playing around with this:
# import math
# from torch.nn import init
# from typing import Callable, Optional

# HookType=Callable[[nn.Module, Tensor, Tensor], Optional[Tensor]]


# class TargetPropLinear(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.in_features=in_features
#         self.out_features=out_features
#         self.F_network=nn.Linear(
#             self.in_features, self.out_features, bias=False)
#         self.G_network=nn.Linear(
#             self.out_features, self.in_features, bias=False)

#     def forward(self, input: Tensor) -> Tensor:
#         return F.linear(input, self.weight, self.bias)

#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )
