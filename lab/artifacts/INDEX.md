# 산출물 인덱스 — 어떤 단계 결과가 어느 파일인지

2026-09-14 재정리: "v3 폴더가 지저분하다"는 지적을 받고, 세 예제 모델(tinycnn_dynamo/
resnet18_dynamo/mobilenet_v2_dynamo)의 파이프라인 결과를 전부
`debug_info_investigation/artifacts/v3/<모델명>/` 밑으로 통합했다. `lab/artifacts/` 밑에는
이제 "버그 수정 전(1차 실행, 비교용)" 결과와 export된 원본 .onnx 파일만 남는다.

## 방법1 — info_track/code_track/join 파이프라인 (5단계) — 정본 위치

세 모델 모두 동일 규칙: `../../../debug_info_investigation/artifacts/v3/<모델명>/`

| 단계 | 파일명 | 설명 |
|---|---|---|
| ① info_track (admit) | `<모델명>.locmap.json` | node.name → {file,line,source,module_scope...} |
| ② code_track (tail 대역) | `<모델명>_kernels.c` | `#line` 삽입 전 데모 C |
| ③ join | `<모델명>_kernels.line.c` | `#line` 투영된 최종 C |
| ④ gcc -g | `<모델명>_kernels_dbg` | DWARF 포함 바이너리 |
| ⑤ 검증 종합 | `<모델명>_pipeline_report.json` | inject_stats + verify.checks(4개) + gdb_probe |

- [../../debug_info_investigation/artifacts/v3/tinycnn_dynamo/](../../debug_info_investigation/artifacts/v3/tinycnn_dynamo/) — 5노드
- [../../debug_info_investigation/artifacts/v3/resnet18_dynamo/](../../debug_info_investigation/artifacts/v3/resnet18_dynamo/) — 49노드
- [../../debug_info_investigation/artifacts/v3/mobilenet_v2_dynamo/](../../debug_info_investigation/artifacts/v3/mobilenet_v2_dynamo/) — 100노드

재실행: `bash lab/run_examples.sh` (세 모델 전부 다시 돌려서 이 위치에 씀. tinycnn용 단독
실행은 `bash debug_info_investigation/run_pipeline.sh` — 둘 다 `--workdir` 생략 시
`run_full_pipeline.py`가 `artifacts/v3/<onnx파일명>/`을 자동으로 만든다.)

## `lab/artifacts/`에 남아있는 것 (여기로 옮기지 않음)

| 경로 | 내용 |
|---|---|
| `resnet18_dynamo.onnx`, `mobilenet_v2_dynamo.onnx`(+`.data`) | `export_complex_models.py`가 만든 원본 export |
| `resnet18/`, `mobilenet_v2/` | **버그 수정 전(1차 실행)** 결과 — "모든 노드가 한 줄로 수렴"하던 버그 재현용 비교 데이터. 파일 구성은 위 표와 동일하지만 폴더 위치만 다름 |
| `resnet18/resnet18_dynamo_onnxir.onnx.mlir`, `mobilenet_v2/mobilenet_v2_dynamo_onnxir.onnx.mlir` | 방법2(onnx-mlir 임포터 패치) 출력 — 방법1과 무관, 옮기지 않음 |
| `tinycnn_recheck_with_patch.onnx.mlir` | 방법2 대조군(tinycnn) 재현용 |

## 방법2 — onnx-mlir 임포터 패치 (커밋 안 됨)

| 모델 | MLIR 출력 (`--EmitONNXIR --mlir-print-debuginfo`) |
|---|---|
| TinyCNN (5노드, 재현용) | [tinycnn_recheck_with_patch.onnx.mlir](tinycnn_recheck_with_patch.onnx.mlir) |
| ResNet-18 (49노드) | [resnet18/resnet18_dynamo_onnxir.onnx.mlir](resnet18/resnet18_dynamo_onnxir.onnx.mlir) |
| MobileNetV2 (100노드) | [mobilenet_v2/mobilenet_v2_dynamo_onnxir.onnx.mlir](mobilenet_v2/mobilenet_v2_dynamo_onnxir.onnx.mlir) |

수치 근거는 [method2_importer_patch_findings.md](method2_importer_patch_findings.md) 참고.
패치 소스 자체는 `src/Builder/FrontendDialectTransformer.cpp` (`git diff`로 확인, 커밋 안 됨).

## 한 파일만 봐야 한다면

- 방법1 최신 결과 요약: `debug_info_investigation/artifacts/v3/<모델명>/<모델명>_pipeline_report.json`
- 방법1 버그 수정 전/후 비교: `lab/artifacts/resnet18/...json` (전) vs `v3/resnet18_dynamo/...json` (후)
- 방법2 결과 확인: 해당 `.onnx.mlir`에서 `grep '^#loc[0-9]* = loc(' <파일>` 실행
- 전체 내러티브(원인 분석, 버그 수정 경위, 폴더 재정리 경위 포함): [../lab.md](../lab.md)
