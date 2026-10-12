# PyTorch ONNX Conversion Report

```
✅ Obtain model graph with `torch.export.export(..., strict=False)`
⚪ Obtain model graph with `torch.export.export(..., strict=True)`
⚪ Obtain model graph with `torch.export.draft_export`
✅ Decompose operators for ONNX compatibility
✅ Translate the graph into ONNX
⚪ Run `onnx.checker` on the ONNX model
⚪ Execute the model with ONNX Runtime
⚪ Validate model output accuracy
```

## Error messages

```pytb
No errors
```

## Exported program

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_fc1_weight: "f32[128, 196]", p_fc1_bias: "f32[128]", p_fc2_weight: "f32[10, 128]", p_fc2_bias: "f32[10]", x: "f32[1, 1, 28, 28]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/pooling.py:224 in forward, code: return F.max_pool2d(
            max_pool2d: "f32[1, 1, 14, 14]" = torch.ops.aten.max_pool2d.default(x, [2, 2], [2, 2]);  x = None
    
            # File: /workdir/onnx-mlir/inspect_export_metadata.py:22 in forward, code: x = x.flatten(1)
            view: "f32[1, 196]" = torch.ops.aten.view.default(max_pool2d, [1, 196]);  max_pool2d = None
    
            # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
            linear: "f32[1, 128]" = torch.ops.aten.linear.default(view, p_fc1_weight, p_fc1_bias);  view = p_fc1_weight = p_fc1_bias = None
    
            # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
            relu: "f32[1, 128]" = torch.ops.aten.relu.default(linear);  linear = None
    
            # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_1: "f32[1, 10]" = torch.ops.aten.linear.default(relu, p_fc2_weight, p_fc2_bias);  relu = p_fc2_weight = p_fc2_bias = None
            return (linear_1,)
    
Graph signature:
    # inputs
    p_fc1_weight: PARAMETER target='fc1.weight'
    p_fc1_bias: PARAMETER target='fc1.bias'
    p_fc2_weight: PARAMETER target='fc2.weight'
    p_fc2_bias: PARAMETER target='fc2.bias'
    x: USER_INPUT
    
    # outputs
    linear_1: USER_OUTPUT
    
Range constraints: {}

```

## ONNX model

```python
<
    ir_version=10,
    opset_imports={'': 20},
    producer_name='pytorch',
    producer_version='2.14.0+cpu',
    domain=None,
    model_version=None,
>
graph(
    name=main_graph,
    inputs=(
        %"x"<FLOAT,[1,1,28,28]>
    ),
    outputs=(
        %"linear_1"<FLOAT,[1,10]>
    ),
    initializers=(
        %"fc1.weight"<FLOAT,[128,196]>{TorchTensor(...)},
        %"fc1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"fc2.weight"<FLOAT,[10,128]>{TorchTensor(...)},
        %"fc2.bias"<FLOAT,[10]>{TorchTensor<FLOAT,[10]>(Parameter containing: tensor([ 0.0860, -0.0659, -0.0124,  0.0681, -0.0069, -0.0244,  0.0678,  0.0328, 0.0767, -0.0242], requires_grad=True), name='fc2.bias')},
        %"val_4"<INT64,[2]>{Tensor<INT64,[2]>(array([  1, 196]), name='val_4')}
    ),
) {
    0 |  # node_max_pool2d
         %"max_pool2d"<FLOAT,[1,1,14,14]> ⬅️ ::MaxPool(%"x") {storage_order=0, dilations=(1, 1), ceil_mode=0, pads=(0, 0, 0, 0), auto_pad='NOTSET', strides=(2, 2), kernel_shape=(2, 2)}
    1 |  # node_view
         %"view"<FLOAT,[1,196]> ⬅️ ::Reshape(%"max_pool2d", %"val_4"{[1, 196]}) {allowzero=1}
    2 |  # node_linear
         %"linear"<FLOAT,[1,128]> ⬅️ ::Gemm(%"view", %"fc1.weight"{...}, %"fc1.bias"{...}) {beta=1.0, transB=1, alpha=1.0, transA=0}
    3 |  # node_relu
         %"relu"<FLOAT,[1,128]> ⬅️ ::Relu(%"linear")
    4 |  # node_linear_1
         %"linear_1"<FLOAT,[1,10]> ⬅️ ::Gemm(%"relu", %"fc2.weight"{...}, %"fc2.bias"{[0.08602125197649002, -0.0658772811293602, -0.01236514002084732, 0.06805441528558731, -0.006939791142940521, -0.024448193609714508, 0.06775873154401779, 0.032767392694950104, 0.07670677453279495, -0.024249285459518433]}) {beta=1.0, transB=1, alpha=1.0, transA=0}
    return %"linear_1"<FLOAT,[1,10]>
}


```

## Analysis

PyTorch ONNX Conversion Analysis

## Model Information

The model has 26506 parameters and 0 buffers (non-trainable parameters).
Number of parameters per dtype:
```python
defaultdict(<class 'int'>, {torch.float32: 26506})
```
Number of buffers per dtype:
```python
defaultdict(<class 'int'>, {})
```

Inputs:
- `x`: `TensorMetadata(shape=torch.Size([1, 1, 28, 28]), dtype=torch.float32, requires_grad=False, stride=(784, 784, 28, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`

Outputs:
- `linear_1`: `TensorMetadata(shape=torch.Size([1, 10]), dtype=torch.float32, requires_grad=False, stride=(10, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`

The FX graph has 11 nodes in total. Number of FX nodes per op:
- `placeholder`: 5
- `call_function`: 5
- `output`: 1


Of the call_function nodes, the counts of operators used are:

- `aten.linear.default`: 2
- `aten.max_pool2d.default`: 1
- `aten.view.default`: 1
- `aten.relu.default`: 1

## ONNX Conversion Information

All operators in the model have registered ONNX decompositions.

## Decomposition comparison

Ops exist only in the ExportedProgram before decomposition: `['aten.flatten.using_ints']`

Ops exist only in the ExportedProgram after decomposition: `['aten.view.default']`

