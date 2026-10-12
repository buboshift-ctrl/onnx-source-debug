#!/usr/bin/env bash
# Reproduces the ONNX provenance / debug-info investigation.
# Env: onnx-mlir build at build/Debug/bin/onnx-mlir, python3 with torch(+cpu),
#      onnx, onnxscript installed. See debug_info_investigation.md for context.
# no `set -e`: step 4 (--enable-debug-info) is expected to fail.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OM="$ROOT/build/Debug/bin/onnx-mlir"
WORK="$HERE/artifacts"
mkdir -p "$WORK"
cd "$WORK"

echo "### 1. Export the same tiny CNN two ways (dynamo=True vs legacy) ###"
python3 "$ROOT/inspect_export_metadata.py" | tee export_metadata.log

echo
echo "### 2. Does onnx-mlir keep node metadata_props / doc_string? ###"
"$OM" --EmitONNXIR --mlir-print-debuginfo --preserveLocations \
      tinycnn_dynamo.onnx -o tinycnn_dynamo_onnxir
grep -nE '^#loc|onnx_node_name|onnx\.(MaxPool|Gemm|Relu|Reshape)' \
      tinycnn_dynamo_onnxir.onnx.mlir | grep -v Constant | tee tinycnn_dynamo_onnxir.locs.txt

echo
echo "### 3. Does loc() survive lowering? ###"
"$OM" --EmitMLIR   --mlir-print-debuginfo --preserveLocations tinycnn_dynamo.onnx -o tinycnn_dynamo_mlir
"$OM" --EmitLLVMIR --mlir-print-debuginfo --preserveLocations tinycnn_dynamo.onnx -o tinycnn_dynamo_llvm
echo "-- loc defs still present at LLVM-dialect level:"
grep -nE '^#loc[0-9]* = loc\(' tinycnn_dynamo_llvm.onnx.mlir | head

echo
echo "### 4. Built-in DWARF path (--enable-debug-info) ###"
"$OM" --EmitObj --enable-debug-info --preserveLocations tinycnn_dynamo.onnx -o tc_di \
  || echo ">>> --enable-debug-info FAILS the MLIR LLVM verifier (expected, see report)"

echo
echo "### 5. Normal artifact + its DWARF origin ###"
"$OM" --EmitLib tinycnn_dynamo.onnx -o tc_lib
readelf -S tc_lib.so | grep -i debug || true
llvm-dwarfdump --debug-info tc_lib.so | grep -E 'DW_AT_name|DW_AT_producer' | head
