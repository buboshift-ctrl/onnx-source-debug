"""
PyTorch가 ONNX로 변환될 때 소스 provenance 정보를 실제로 얼마나 남기는지 확인.
도커(onnx-mlir)가 아니라 PyTorch가 설치된 환경(로컬/venv)에서 실행할 것.

    pip install torch onnx --break-system-packages   # 필요하면
    python3 inspect_export_metadata.py
"""
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(196, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = TinyCNN().eval()
dummy = torch.randn(1, 1, 28, 28)

# 1) 신식 exporter (dynamo=True) - 이게 요즘 권장 경로
onnx_program = torch.onnx.export(
    model, (dummy,), "tinycnn_dynamo.onnx", dynamo=True, report=True
)

import onnx  # noqa: E402

m = onnx.load("tinycnn_dynamo.onnx")
print("=== dynamo=True exporter 결과 ===")
for node in m.graph.node:
    print(f"\n[node] {node.name}  ({node.op_type})")
    print(f"  doc_string: {node.doc_string!r}")
    if node.metadata_props:
        for kv in node.metadata_props:
            print(f"  metadata_props[{kv.key}] = {kv.value}")
    else:
        print("  metadata_props: (없음)")

# 2) 구식 exporter (dynamo=False, 기본 doc_string 제거 여부 확인용 비교)
torch.onnx.export(model, (dummy,), "tinycnn_legacy.onnx", dynamo=False)
m2 = onnx.load("tinycnn_legacy.onnx")
print("\n\n=== dynamo=False(구식) exporter 결과 ===")
for node in m2.graph.node:
    print(f"[node] {node.name} ({node.op_type})  doc_string={node.doc_string!r}  "
          f"metadata_props={len(node.metadata_props)}개")
