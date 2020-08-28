import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda


class CorrelationFunction(Function):

    def __init__(self):
        super(CorrelationFunction, self).__init__()
        # self.pad_size = pad_size
        # self.kernel_size = kernel_size
        # self.max_displacement = max_displacement
        # self.stride1 = stride1
        # self.stride2 = stride2
        # self.corr_multiply = corr_multiply
        # # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx, input1, input2, pad, kernel, max_d, stride1, stride2, corr_m):
        ctx.save_for_backward(input1, input2)
        ctx.arg = pad, kernel, max_d, stride1, stride2, corr_m

        rbot1 = torch.empty_like(input1)
        rbot2 = torch.empty_like(input2)
        output = torch.empty_like(input1)

        correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                 pad, kernel, max_d, stride1, stride2, corr_m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        pad, kernel, max_d, stride1, stride2, corr_m = ctx.arg

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      pad, kernel, max_d, stride1, stride2, corr_m)

        return grad_input1, grad_input2


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction().apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement,
                                             self.stride1, self.stride2, self.corr_multiply)

        return result
