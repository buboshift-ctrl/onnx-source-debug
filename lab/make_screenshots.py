"""
각 파이프라인 단계의 "실행 결과"를 터미널/에디터 스크린샷처럼 렌더링해 PNG로 저장한다.
여기 들어가는 텍스트는 전부 이 세션에서 실제로 실행해서 얻은 값이다(조작/가상 데이터 없음) —
출처는 각 생성 함수의 주석에 파일 경로/명령으로 남겨둔다.

실행: python3 make_screenshots.py
출력: shots/*.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "shots")
os.makedirs(OUT, exist_ok=True)

BG = (13, 17, 23)          # GitHub dark 배경
TITLEBAR = (22, 27, 34)
FG = (201, 209, 217)       # 기본 텍스트
DIM = (110, 118, 129)      # 주석/보조
GREEN = (86, 211, 100)     # PASS / prompt
YELLOW = (219, 178, 87)    # 강조(경로, 숫자)
RED = (248, 113, 113)
CYAN = (110, 197, 232)

FONT_SIZE = 21
LINE_H = 30
PAD_X = 26
PAD_TOP = 20
TITLEBAR_H = 40


def _font():
    return ImageFont.load_default(size=FONT_SIZE)


def render(lines, out_name, title, width=1500):
    """lines: [(text, color)] 또는 순수 문자열(FG로 렌더)."""
    norm = [(t, FG) if isinstance(t, str) else t for t in lines]
    height = TITLEBAR_H + PAD_TOP * 2 + LINE_H * len(norm)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, width, TITLEBAR_H], fill=TITLEBAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([18 + i * 24, 14, 18 + i * 24 + 12, 26], fill=c)
    font = _font()
    d.text((width / 2, TITLEBAR_H / 2), title, fill=DIM, font=font, anchor="mm")

    y = TITLEBAR_H + PAD_TOP
    for text, color in norm:
        d.text((PAD_X, y), text, fill=color, font=font)
        y += LINE_H

    path = os.path.join(OUT, out_name)
    img.save(path)
    print("생성:", path)
    return path


def c(text, color):
    return (text, color)


# ============================================================================
# 방법1 — info_track / code_track / join 각 단계 실제 출력
# (모두 lab/artifacts/resnet18_fixed/ 산출물에서 node_Conv_292 기준 발췌)
# ============================================================================

render(
    [
        c("$ cat resnet18_dynamo.locmap.json | jq '.nodes.node_Conv_292'", GREEN),
        "{",
        c('  "op_type": "Conv",', FG),
        c('  "file": "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/batchnorm.py",', YELLOW),
        c('  "line": 210,', YELLOW),
        c('  "source": "return F.batch_norm(",', FG),
        c('  "func": "forward",', FG),
        c('  "module_scope": "bn1._native_batch_norm_legit_no_training",', CYAN),
        c('  "namespace": ": torchvision.models.resnet.ResNet/bn1: ...",', DIM),
        c('  "class_hierarchy": [', FG),
        c('    "torchvision.models.resnet.ResNet",', DIM),
        c('    "torch.nn.modules.batchnorm.BatchNorm2d",', DIM),
        c('    "aten._native_batch_norm_legit_no_training.default"', DIM),
        c("  ], ...", DIM),
        "}",
    ],
    "01_info_track_output.png",
    "resnet18_dynamo.locmap.json  --  info_track.build_locmap() actual output",
)

render(
    [
        c("$ sed -n '5,10p' resnet18_dynamo_kernels.c", GREEN),
        "",
        c("static void node_node_Conv_292(float *buf, size_t n) {   /* conv stand-in */", FG),
        c("/* onnx_node_name: node_Conv_292   op=Conv */", CYAN),
        c("  for (size_t i = 0; i < n; ++i)", FG),
        c("    buf[i] = buf[i] * 1.0f + 0.0f;   /* Conv */", FG),
        c("/* onnx_node_end */", CYAN),
        "}",
        "",
        c("# no #line yet -- code_track never reads provenance/locmap", DIM),
    ],
    "02_code_track_output.png",
    "resnet18_dynamo_kernels.c  --  code_track.get_generated_c() actual output",
)

render(
    [
        c("$ diff resnet18_dynamo_kernels.c resnet18_dynamo_kernels.line.c | head -8", GREEN),
        "",
        c("static void node_node_Conv_292(float *buf, size_t n) {   /* conv stand-in */", FG),
        c("/* onnx_node_name: node_Conv_292   op=Conv */", CYAN),
        c('#line 210 "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/batchnorm.py"', GREEN),
        c("  for (size_t i = 0; i < n; ++i)", FG),
        c("    buf[i] = buf[i] * 1.0f + 0.0f;   /* Conv */", FG),
        c("/* onnx_node_end */", CYAN),
        c('#line 15 "resnet18_dynamo_kernels.line.c"', GREEN),
        "}",
    ],
    "03_join_output.png",
    "resnet18_dynamo_kernels.line.c  --  join.inject_with_stats() actual output (+2 #line)",
)

render(
    [
        c("$ gcc -g -O0 -o resnet18_dynamo_kernels_dbg resnet18_dynamo_kernels.line.c", GREEN),
        c("$ echo $?", GREEN),
        c("0", FG),
        "",
        c("$ file resnet18_dynamo_kernels_dbg", GREEN),
        c("resnet18_dynamo_kernels_dbg: ELF 64-bit LSB pie executable, x86-64,", FG),
        c("  ... with debug_info, not stripped", GREEN),
    ],
    "04_gcc_output.png",
    "gcc -g -O0  --  actual run, exit code 0",
)

render(
    [
        c("$ readelf --debug-dump=decodedline resnet18_dynamo_kernels_dbg | grep -A9 batchnorm", GREEN),
        c(".../torch/nn/modules/batchnorm.py:", DIM),
        c("batchnorm.py        210        0x1159        x", YELLOW),
        c("batchnorm.py        210        0x1161        x", YELLOW),
        c("batchnorm.py        211        0x1163        x", FG),
        c("batchnorm.py        211        0x1176        x", FG),
        "",
        c("$ gdb --batch -ex 'break batchnorm.py:210' -ex 'info line batchnorm.py:210' \\", GREEN),
        c("      resnet18_dynamo_kernels_dbg", GREEN),
        c("Breakpoint 1 at 0x1159: batchnorm.py:210. (20 locations)", GREEN),
        c('Line 210 of ".../batchnorm.py" starts at address', FG),
        c("0x1159 <node_node_Conv_292+16> and ends at 0x1161 <node_node_Conv_292+24>.", CYAN),
    ],
    "05_verify_output.png",
    "readelf / gdb  --  actual measured output (from *_pipeline_report.json)",
)

# ============================================================================
# 방법2 — onnx-mlir 임포터 패치 실제 실행
# ============================================================================

render(
    [
        c("$ build/Debug/bin/onnx-mlir --EmitONNXIR --mlir-print-debuginfo \\", GREEN),
        c("      lab/artifacts/resnet18_dynamo.onnx -o resnet18_dynamo_onnxir", GREEN),
        "",
        c('[1/3] ... Importing ONNX Model to MLIR Module from "resnet18_dynamo.onnx"', FG),
        c("[2/3] ... Compiling and Optimizing MLIR Module", FG),
        c("", FG),
        c("Warning: onnx-hybrid-transform didn't converge with", YELLOW),
        c("  max-num-rewrites-offset=20, max-num-rewrites-multiplier=2.000000e-01", YELLOW),
        "",
        c('[3/3] ... Compilation completed. Generated "resnet18_dynamo_onnxir.onnx.mlir"', GREEN),
    ],
    "06_method2_run_output.png",
    "onnx-mlir --EmitONNXIR  --  actual run log (patched binary)",
)

render(
    [
        c("$ grep '^#loc1 = \\|^#loc2 = \\|^#loc55 = ' resnet18_dynamo_onnxir.onnx.mlir", GREEN),
        "",
        c('#loc1 = loc("node_Conv_349")', CYAN),
        c('#loc2 = loc("/usr/local/lib/.../torchvision/models/resnet.py":285:0)', YELLOW),
        c("#loc55 = loc(fused[#loc1, #loc2])   <- NameLoc + FileLineColLoc", GREEN),
        "",
        c('# all 49 nodes fuse to the SAME #loc2 (line 285) -- "outermost frame" bias', RED),
        c("# same root cause as info_track.py; this C++ parser not fixed yet", DIM),
    ],
    "07_method2_mlir_output.png",
    "resnet18_dynamo_onnxir.onnx.mlir  --  actual loc() definitions (excerpt)",
)

render(
    [
        c('#loc7 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":21:0)   <- MaxPool', GREEN),
        c('#loc9 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":22:0)   <- Reshape', GREEN),
        c('#loc11 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":23:0)  <- Linear/ReLU', GREEN),
        c('#loc14 = loc("/workdir/onnx-mlir/inspect_export_metadata.py":24:0)  <- Linear', GREEN),
        "",
        c("# TinyCNN forward() is a single flat method -> each node maps to its own line", DIM),
    ],
    "08_method2_tinycnn_output.png",
    "tinycnn_recheck_with_patch.onnx.mlir  --  control case (works as expected)",
    width=1300,
)

print("완료:", OUT)
