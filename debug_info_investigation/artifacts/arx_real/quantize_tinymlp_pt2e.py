"""
MaxPool/Reshape 경계 양자화 이슈를 피하기 위해, Linear/ReLU만으로 구성된 더 단순한 모델로
PT2E PTQ + dynamo export를 검증한다. 목적은 동일: 양자화된 노드에서도 stack_trace가
살아남고, 실제 ARX 파이프라인이 .py:줄까지 이어주는지 확인.
"""
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(196, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = TinyMLP().eval()
dummy = torch.randn(1, 196)

print("=== 1) torch.export ===")
exported = torch.export.export(model, (dummy,))
print("성공")

print("=== 2) 양자화 준비 ===")
quantizer = X86InductorQuantizer()
quantizer.set_global(get_default_x86_inductor_quantization_config())
prepared = prepare_pt2e(exported.module(), quantizer)

print("=== 3) 캘리브레이션 ===")
with torch.no_grad():
    for _ in range(10):
        prepared(torch.randn(1, 196))

print("=== 4) 변환 ===")
quantized = convert_pt2e(prepared)
print(quantized.graph)

print("\n=== 5) dynamo export ===")
try:
    torch.onnx.export(
        quantized, (dummy,), "/workdir/external/arx-build/tinymlp_quant_dynamo.onnx",
        dynamo=True, report=False,
    )
    print("export 성공")
except Exception as e:
    print("export 실패:", repr(e))
