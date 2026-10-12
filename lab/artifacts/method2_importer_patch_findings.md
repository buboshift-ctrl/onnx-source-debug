# 방법2(onnx-mlir 임포터 패치) — 실측 수치 기록

`src/Builder/FrontendDialectTransformer.cpp`의 커밋 안 된 `ImportLoc()`/`ParseStackTrace()`
패치(2026-09-10 빌드 바이너리에 반영됨)를 3개 모델에 `--EmitONNXIR --mlir-print-debuginfo`로
돌려 얻은 수치. 대화 중 터미널 출력으로만 보여줬던 값을 파일로 남긴다.

측정 시각: 2026-09-14 (이 세션). 재현 명령:
```bash
build/Debug/bin/onnx-mlir --EmitONNXIR --mlir-print-debuginfo <model>.onnx -o <out>
grep -c 'loc(#loc\|loc("' <out>.onnx.mlir            # loc(...) 총 등장 수
grep -c '^#loc[0-9]* = loc(' <out>.onnx.mlir         # #locN 정의 수
grep -c '\.py":[0-9]*:[0-9]*)' <out>.onnx.mlir       # FileLineColLoc(.py:N:N) 형태 수
```

## 수치

| 모델 | ONNX 노드 수 | loc(...) 총 등장 | #locN 정의 수 | FileLineColLoc(.py:N:N) 개수 |
|---|---|---|---|---|
| TinyCNN | 5 | (참고: 정의 20개, 아래 목록) | 20 | 4 (노드마다 서로 다른 줄) |
| ResNet-18 | 49 | 201 | 104 | 1 (**모든 노드가 같은 줄로 수렴** — 원인은 아래) |
| MobileNetV2 | 100 | 376 | 224 | 1 (동일 현상) |

## TinyCNN — 노드별로 정확히 다른 줄 (기대한 동작)

```
#loc = loc(unknown)
#loc1 = loc("Initializer_fc1.weight")
...
#loc6 = loc("node_max_pool2d")
#loc7 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":21:0)
#loc8 = loc("node_view")
#loc9 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":22:0)
#loc10 = loc("node_linear")
#loc11 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":23:0)
#loc12 = loc("node_relu")
#loc13 = loc("node_linear_1")
#loc14 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":24:0)
#loc15 = loc(fused[#loc6, #loc7])
#loc16 = loc(fused[#loc8, #loc9])
#loc17 = loc(fused[#loc10, #loc11])
#loc18 = loc(fused[#loc12, #loc11])   # relu와 linear가 같은 소스 줄(23) 공유 — 정상(같은 줄 코드)
#loc19 = loc(fused[#loc13, #loc14])
```
→ MaxPool=21, Reshape=22, Linear+ReLU=23, Linear_1=24 — 원본 4줄에 정확히 대응.

## ResNet-18 — "공용 헬퍼" 효과로 전부 한 줄 수렴 (원인 규명됨)

```
#loc1 = loc("node_Conv_349")
#loc2 = loc("/usr/local/lib/python3.12/dist-packages/torchvision/models/resnet.py":285:0)
...
#loc55 = loc(fused[#loc1, #loc2])   # Conv_349
#loc56 = loc(fused[#loc3, #loc2])   # 다른 노드도 전부 #loc2(285줄)
#loc57 = loc(fused[#loc4, #loc2])
#loc58 = loc(fused[#loc5, #loc2])
#loc59 = loc(fused[#loc6, #loc2])
```
49개 노드 전부가 `#loc2`(resnet.py:285, `return self._forward_impl(x)`) 하나에 fuse됨.
MobileNetV2도 100개 전부 mobilenetv2.py:174 하나로 동일하게 수렴.

**원인**: `ParseStackTrace()`가 `pkg.torch.onnx.stack_trace` 문자열에서 처음 만나는
`File "…", line N`만 취하는데, 이게 파이썬 traceback 포맷상 "가장 바깥쪽 호출 프레임"이다.
`ResNet.forward()` → `_forward_impl()` → `BasicBlock.forward()`처럼 forward가 여러 겹인
모델에서는, 어떤 연산이 실행되든 가장 바깥쪽 프레임은 항상 동일한 한 줄
(`return self._forward_impl(x)`)이다. 파이썬 쪽 동형 로직(`info_track.py`의
`_pick_user_frame`)에서 같은 원인·같은 증상을 먼저 확인했고(과제 PPT 참고), 그쪽은
"가장 안쪽(마지막) 프레임 우선"으로 수정·검증을 마쳤다. 이 C++ `ParseStackTrace()`는
**아직 그 수정을 반영하지 않은 상태**(사용자 지시로 이번 세션에서는 손대지 않음, 논문 초안
자료로만 정리).

## 남은 일

- `ParseStackTrace()`가 문자열의 첫 프레임이 아니라 **마지막 프레임**(또는 프레임워크 마커로
  걸러진 가장 안쪽 프레임)을 선택하도록 보완 — `info_track.py`의 수정과 대칭.
- 보완 후 동일 3개 모델로 재측정해서 이 파일에 "수정 후" 표를 추가.
- `FusedLoc`이 `--EmitMLIR`/`--EmitLLVMIR`까지 생존하는지는 별도 조사 필요(3순위 항목).
