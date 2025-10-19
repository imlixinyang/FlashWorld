import gc
from typing import Tuple
import copy
import torch
import tqdm


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    """Quantizes a floating-point tensor to FP8 (E4M3 format) using static scaling.
    
    Performs uniform quantization of the input tensor by:
    1. Scaling the tensor values using the provided inverse scale factor
    2. Clamping values to the representable range of FP8 E4M3 format
    3. Converting to FP8 data type
    
    Args:
        tensor (torch.Tensor): Input tensor to be quantized (any floating-point dtype)
        inv_scale (float): Inverse of the quantization scale factor (1/scale)
                         (Must be pre-calculated based on tensor statistics)
    
    Returns:
        torch.Tensor: Quantized tensor in torch.float8_e4m3fn format
    
    Note:
        - Uses the E4M3 format (4 exponent bits, 3 mantissa bits, no infinity/nan)
        - This is a static quantization (scale factor must be pre-determined)
        - For dynamic quantization, see per_tensor_quantize()
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype, native_fp8_support=False):
    """Performs FP8 GEMM (General Matrix Multiplication) operation with optional native hardware support.
    Args:
        A (torch.Tensor): Input tensor A (FP8 or other dtype)
        A_scale (torch.Tensor/float): Scale factor for tensor A
        B (torch.Tensor): Input tensor B (FP8 or other dtype)
        B_scale (torch.Tensor/float): Scale factor for tensor B
        bias (torch.Tensor/None): Optional bias tensor
        out_dtype (torch.dtype): Output data type
        native_fp8_support (bool): Whether to use hardware-accelerated FP8 operations
        
    Returns:
        torch.Tensor: Result of GEMM operation 
    """
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output = torch._scaled_mm(
            A_input,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )

    return output

def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1:]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


# Class responsible for quantizing weights
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
            self,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.nn.Parameter,
            native_fp8_support: bool = False,
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = torch.nn.Parameter(bias.to(dtype), requires_grad=False)
        self.native_fp8_support = native_fp8_support
        self.dtype = dtype

    # @torch.compile
    def forward(self, x):
        if x.dtype !=self.dtype:
            x = x.to(self.dtype)
        qinput, x_scale = per_tensor_quantize(x)
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
        )
        return output


def FluxFp8GeMMProcessor(model: torch.nn.Module):
    """Processes a PyTorch model to convert eligible Linear layers to FP8 precision.
    
    This function performs the following operations:
    1. Checks for native FP8 support on the current GPU
    2. Identifies target Linear layers in transformer blocks
    3. Quantizes weights to FP8 format
    4. Replaces original Linear layers with FP8DynamicLinear versions
    5. Performs memory cleanup
    
    Args:
        model (torch.nn.Module): The neural network model to be processed.
                                Should contain transformer blocks with Linear layers.
    """
    native_fp8_support = (
            torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
    )
    named_modules = list(model.named_modules())
    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights to fp8"):
        if isinstance(linear, torch.nn.Linear) and "blocks" in name:
            quant_weight, weight_scale = per_tensor_quantize(linear.weight)
            bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
            quant_linear = FP8DynamicLinear(
                weight=quant_weight,
                weight_scale=weight_scale,
                bias=bias,
                native_fp8_support=native_fp8_support,
            )
            replace_module(model, name, quant_linear)
            del linear.weight
            del linear.bias
            del linear
    cleanup_memory()