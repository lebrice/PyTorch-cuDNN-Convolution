from torch.nn import ConvTranspose2d
from custom_conv_transpose2d_function import custom_conv_transpose2d


class CustomConvTranspose2d(ConvTranspose2d):
    def forward(self, input, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )

        return custom_conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
