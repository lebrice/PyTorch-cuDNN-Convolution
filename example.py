import torch
from torch.utils.cpp_extension import load

# load the PyTorch extension
cudnn_convolution = load(
    name="cudnn_convolution", sources=["cudnn_convolution.cpp"], verbose=True
)
B = 2
# create dummy x, convolutional weights and bias
x = torch.zeros(B, 3, 32, 32).to("cuda")
weight = torch.zeros(64, 3, 5, 5).to("cuda")
bias = torch.zeros(64).to("cuda")

stride = (2, 2)
padding = (0, 0)
dilation = (1, 1)
groups = 1

# compute the result of convolution
output = cudnn_convolution.convolution(
    x, weight, bias, stride, padding, dilation, groups, False, False
)

# create dummy gradient w.r.t. the output
grad_output = torch.zeros(B, 64, 14, 14).to("cuda")

# compute the gradient w.r.t. the weights and x
grad_weight = cudnn_convolution.convolution_backward_weight(
    x, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False
)
grad_input = cudnn_convolution.convolution_backward_input(
    x.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False
)

print(grad_weight.shape)
print(grad_input.shape)

in_channels = 1
out_channels = 1
kernel_size = k = 3
# create dummy x, transpose-convolutional weights and bias
x = torch.rand(B, in_channels, 4, 4, requires_grad=True, device="cuda")
weight = torch.rand(out_channels, in_channels, k, k, requires_grad=True, device="cuda")
bias = torch.rand(out_channels, requires_grad=True, device="cuda")


stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)
output_padding = (0, 0)
groups = 1


from torch.nn.functional import conv_transpose2d

expected_output = conv_transpose2d(
    input=x,
    weight=weight,
    bias=bias,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    groups=groups,
    dilation=dilation,
)

# create dummy gradient w.r.t. the output
grad_output = torch.zeros(B, 1, 6, 6).to("cuda")

# Note: use 'retain_graph=True' so that we can backpropagate again below
expected_output.backward(grad_output, retain_graph=True)
expected_grad_weight = weight.grad
expected_grad_input = x.grad

assert expected_output.requires_grad
assert expected_grad_weight is not None
assert expected_grad_input is not None


##################
## Option 1: Using CPP extension directly (very bad):
##################

from torch import conv_transpose2d

cudnn_transpose_convolution = load(
    name="cudnn_convolution_transpose",
    sources=["cudnn_convolution_transpose.cpp"],
    verbose=True,
)
# compute the result of convolution
output = cudnn_transpose_convolution.convolution_transpose(
    x,
    weight,
    bias,
    padding,
    output_padding,
    stride,
    dilation,
    groups,
    False,
    False,
    # NOTE: Seems like we can't pass keyword arguments ?
    # input=x,
    # weight=weight,
    # bias=bias,
    # padding=padding,
    # output_padding=output_padding,
    # stride=stride,
    # dilation=dilation,
    # groups=groups,
    # benchmark=False,
    # deterministic=False,
)

assert output.shape == expected_output.shape
assert (output == expected_output).all()


# compute the gradient w.r.t. the weights and x
grad_weight = cudnn_transpose_convolution.convolution_transpose_backward_weight(
    weight.shape,
    grad_output,
    x,
    padding,
    stride,
    dilation,
    groups,
    False,
    False,
    False,
)

assert grad_weight.shape == expected_grad_weight.shape
assert (grad_weight == expected_grad_weight).all()


grad_input = cudnn_transpose_convolution.convolution_transpose_backward_input(
    grad_output, weight, padding, stride, dilation, groups, False, False, False,
)

assert grad_input.shape == expected_grad_input.shape
assert (grad_input == expected_grad_input).all()

# -------------------------------
# Option 2: Using a custom pytorch 'Function' (much better, not perfect)
# -------------------------------
# NOTE: Reset the Grad attribute
# x.grad = None
# weight.grad = None
# bias.grad = None
# NOTE: Re-creating the inputs and expected outputs, since we'll manipulate their .grad
# attributes.
x = torch.rand(B, in_channels, 4, 4, requires_grad=True, device="cuda")
weight = torch.rand(out_channels, in_channels, k, k, requires_grad=True, device="cuda")
bias = torch.rand(out_channels, requires_grad=True, device="cuda")

expected_output = conv_transpose2d(
    input=x,
    weight=weight,
    bias=bias,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    groups=groups,
    dilation=dilation,
)

# create dummy gradient w.r.t. the output
grad_output = torch.zeros(B, 1, 6, 6).to("cuda")
expected_output.backward(grad_output)
expected_grad_weight = weight.grad
expected_grad_input = x.grad

from custom_conv_transpose2d_function import custom_conv_transpose2d

output = custom_conv_transpose2d(
    x, weight, bias, stride, padding, output_padding, groups, dilation,
)
output.backward(grad_output, retain_graph=True)
grad_input = x.grad
grad_weight = weight.grad

assert output.shape == expected_output.shape
assert (output == expected_output).all()

assert grad_weight.shape == expected_grad_weight.shape
assert (grad_weight == expected_grad_weight).all()


assert grad_input.shape == expected_grad_input.shape
assert (grad_input == expected_grad_input).all()


##############
# Option 3: Use a subclass of ConvTranspose2D (best)
##############
x = torch.rand(B, in_channels, 4, 4, requires_grad=True, device="cuda")
weight = torch.rand(out_channels, in_channels, k, k, requires_grad=True, device="cuda")
bias = torch.rand(out_channels, requires_grad=True, device="cuda")
from torch.nn import ConvTranspose2d

reference_module = ConvTranspose2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    bias=True,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    groups=groups,
    dilation=1,
).to("cuda")
reference_module.weight.data = weight.data
reference_module.bias.data = bias.data

expected_output = reference_module(x)

# Note: use 'retain_graph=True' so that we can backpropagate again below
expected_output.backward(grad_output, retain_graph=True)
expected_grad_weight = reference_module.weight.grad
expected_grad_bias = reference_module.bias.grad
expected_grad_input = x.grad

assert expected_output.requires_grad
assert expected_grad_weight is not None
assert expected_grad_bias is not None
assert expected_grad_input is not None

from custom_conv_transpose2d_module import CustomConvTranspose2d

module = CustomConvTranspose2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    bias=True,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    groups=groups,
    dilation=1,
).to("cuda")
# module.train()
module.load_state_dict(reference_module.state_dict())

assert (module.weight == reference_module.weight).all()
assert (module.bias == reference_module.bias).all()

output = module(x)
output.backward(grad_output, retain_graph=True)
assert module.weight.requires_grad
grad_input = x.grad
grad_weight = module.weight.grad
grad_bias = module.bias.grad

assert grad_weight is not None
assert output.shape == expected_output.shape
assert (output == expected_output).all()

assert grad_weight.shape == expected_grad_weight.shape
assert (grad_weight == expected_grad_weight).all()

assert grad_input.shape == expected_grad_input.shape
assert (grad_input == expected_grad_input).all()
