import torch
import torch.nn as nn

class TinyDebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        # 12번째 줄에서 연산 실행
        out = self.fc(x) 
        return out

model = TinyDebugModel()
x = torch.randn(1, 4)

# FX Graph Tracing을 뜸 (stack_trace에 testcase.py:12 정보 자동 캡처됨)
traced = torch.fx.symbolic_trace(model)

for node in traced.graph.nodes:
    if node.op == 'call_module':
        # FX 노드 메타데이터에 소스코드 파일/라인 정보가 담김
        print(f"Node: {node.name} -> Target: {node.target}, Stack Trace: {node.meta.get('stack_trace')}")

# ONNX Dynamo Exporter 사용 시 해당 stack_trace 정보가 ONNX node location으로 전이됨
onnx_program = torch.onnx.dynamo_export(model, x)
onnx_program.save("testcase_dynamo.onnx")