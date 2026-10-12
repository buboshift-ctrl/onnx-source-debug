#!/usr/bin/env bash
# (v3) 얇은 셸 — run_full_pipeline.py 오케스트레이터를 부른다.
#   info_track.build_locmap → code_track.get_generated_c → join.inject → gcc -g → readelf/gdb → 3회 재현성
# 전제: build/Debug/bin/onnx-mlir, python3 + torch(+cpu)/onnx/onnxscript, gcc/gdb/readelf.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OM="$ROOT/build/Debug/bin/onnx-mlir"
ONNX="${1:-$HERE/artifacts/tinycnn_dynamo.onnx}"
cd "$HERE"   # info_track/join/code_track 를 import 할 수 있도록

if [ ! -f "$ONNX" ]; then
  echo "dynamo=True ONNX 가 없습니다: $ONNX"
  echo "  python3 $ROOT/inspect_export_metadata.py  로 먼저 tinycnn_dynamo.onnx 를 만드세요."
  exit 2
fi
STEM="$(basename "${ONNX%.onnx}")"
WORKDIR="$HERE/artifacts/v3/$STEM"   # 모델별 폴더로 자동 분리 (2026-09-14, run_full_pipeline.py 기본값과 동일 규칙)
mkdir -p "$WORKDIR"

echo "### 조인 키 사전점검 — onnx-mlir onnx_node_name ↔ locmap 키 ###"
"$OM" --EmitONNXIR --do-not-emit-full-mlir-code "$ONNX" -o "$WORKDIR/${STEM}_joincheck" >/dev/null 2>&1 \
  && python3 - "$ONNX" "$WORKDIR/${STEM}_joincheck.tmp" <<'PY'
import sys, re, json
import info_track
onnx_path, tmp = sys.argv[1], sys.argv[2]
mlir_names = sorted(set(re.findall(r'onnx_node_name = "([^"]+)"', open(tmp).read())))
locmap_keys = sorted(info_track.build_locmap(onnx_path)["nodes"])
print("OK: 일치" if mlir_names == locmap_keys else f"MISMATCH\n  mlir={mlir_names}\n  locmap={locmap_keys}")
PY

echo
echo "### 파이프라인 실행 ###"
exec python3 "$HERE/run_full_pipeline.py" "$ONNX" --workdir "$WORKDIR" --relative-to "$ROOT"
