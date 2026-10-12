"""
tinycnn(5 노드)보다 레이어/연산이 훨씬 많은 실전 모델 2개를
torch.onnx.export(dynamo=True)로 내보내서, info_track/code_track/join
파이프라인이 더 큰 그래프에서도 동작하는지 검증하기 위한 export 스크립트.

대상: torchvision resnet18, mobilenet_v2 (모두 사전학습 가중치 없이 구조만 사용 — 순수 forward
그래프/provenance 메타데이터 확인이 목적이라 랜덤 초기화로 충분).

산출: lab/artifacts/resnet18_dynamo.onnx, lab/artifacts/mobilenet_v2_dynamo.onnx
"""
import os

import torch
from torchvision.models import mobilenet_v2, resnet18

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)

dummy = torch.randn(1, 3, 224, 224)

targets = [
    ("resnet18", resnet18(weights=None).eval()),
    ("mobilenet_v2", mobilenet_v2(weights=None).eval()),
]

for name, model in targets:
    out_path = os.path.join(OUT, f"{name}_dynamo.onnx")
    print(f"=== exporting {name} (dynamo=True) -> {out_path} ===")
    torch.onnx.export(model, (dummy,), out_path, dynamo=True, report=False)

    import onnx

    m = onnx.load(out_path)
    n_nodes = len(m.graph.node)
    n_with_trace = sum(
        1
        for node in m.graph.node
        if any(kv.key == "pkg.torch.onnx.stack_trace" for kv in node.metadata_props)
    )
    print(f"  node_count={n_nodes}  nodes_with_stack_trace={n_with_trace}")
