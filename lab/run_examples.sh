#!/usr/bin/env bash
# 예제 모델 3개(tinycnn_dynamo / resnet18_dynamo / mobilenet_v2_dynamo)를 전부
# info_track → code_track → join → gcc -g → readelf/gdb 검증 파이프라인으로 돌리는 스크립트.
#
# 산출물은 전부 debug_info_investigation/artifacts/v3/<모델명>/ 밑에 모인다 — tinycnn만 있던
# 기존 관례(debug_info_investigation/run_pipeline.sh)를 그대로 따라 세 모델 다 같은 규칙으로
# 저장한다(2026-09-14, "v3가 지저분하다"는 지적을 받고 lab/artifacts/<model>_fixed/ 를
# 이 위치로 통합했다). "버그 수정 전(1차 실행)" 결과는 비교용으로 lab/artifacts/resnet18/,
# lab/artifacts/mobilenet_v2/ 에 그대로 남아있다(이 스크립트가 건드리지 않음).
#
# 하는 일:
#   1) tinycnn_dynamo.onnx 는 이미 있어야 함(inspect_export_metadata.py 로 미리 생성).
#      resnet18/mobilenet_v2 는 export_complex_models.py 로 dynamo=True export.
#   2) 각 모델을 debug_info_investigation/run_full_pipeline.py 로 돌림
#      (--workdir 을 안 주면 파이썬 스크립트가 artifacts/v3/<stem>/ 을 자동으로 씀)
#   3) 세 모델의 verify.ok / matched / 재현성 결과를 요약 출력
#
# 전제: build/Debug/bin/onnx-mlir, python3 + torch(+cpu)/torchvision(+cpu)/onnx, gcc/gdb/readelf.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DII="$ROOT/debug_info_investigation"
PIPE="$DII/run_full_pipeline.py"

TINYCNN_ONNX="$DII/artifacts/tinycnn_dynamo.onnx"
if [ ! -f "$TINYCNN_ONNX" ]; then
  echo "tinycnn_dynamo.onnx 가 없습니다: $TINYCNN_ONNX"
  echo "  python3 $ROOT/inspect_export_metadata.py  로 먼저 만드세요."
  exit 2
fi

echo "### 1) resnet18/mobilenet_v2 export (torch.onnx.export dynamo=True) ###"
cd "$HERE"
python3 export_complex_models.py
echo

# stem -> onnx 경로
declare -A ONNX_OF=(
  [tinycnn_dynamo]="$TINYCNN_ONNX"
  [resnet18_dynamo]="$HERE/artifacts/resnet18_dynamo.onnx"
  [mobilenet_v2_dynamo]="$HERE/artifacts/mobilenet_v2_dynamo.onnx"
)

overall_ok=1
cd "$DII"   # info_track/code_track/join 를 import 할 수 있도록
for STEM in tinycnn_dynamo resnet18_dynamo mobilenet_v2_dynamo; do
  ONNX="${ONNX_OF[$STEM]}"
  WORKDIR="$DII/artifacts/v3/$STEM"
  echo "### 2) 파이프라인 실행 — $STEM  (workdir: $WORKDIR) ###"
  python3 "$PIPE" "$ONNX" --workdir "$WORKDIR" --relative-to "$ROOT"
  rc=$?
  if [ $rc -ne 0 ]; then
    overall_ok=0
    echo ">>> $STEM: 파이프라인 실패 (exit $rc)"
  fi
  echo
done

echo "### 요약 ###"
for STEM in tinycnn_dynamo resnet18_dynamo mobilenet_v2_dynamo; do
  REPORT="$DII/artifacts/v3/$STEM/${STEM}_pipeline_report.json"
  python3 - "$STEM" "$REPORT" <<'PY'
import json, sys
model, path = sys.argv[1], sys.argv[2]
r = json.load(open(path))
v = r["verify"]
matched = len(r["inject_stats"]["matched"])
unknown = len(r["inject_stats"]["unknown"])
print(f"{model}: verify.ok={v['ok']}  matched={matched} unknown={unknown}  "
      f"checks={[c['ok'] for c in v['checks']]}")
PY
done

exit $((1 - overall_ok))
