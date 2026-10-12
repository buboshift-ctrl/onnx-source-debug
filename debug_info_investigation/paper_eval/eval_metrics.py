#!/usr/bin/env python3
"""
논문 평가용 메트릭 추출 스크립트.

각 모델별로:
1. 기본 지표 (노드 수, 위치 정보 부착 수 등)
2. 매핑 정확도 (op_type vs 선택된 stack frame 일치도)
3. 오버헤드 (크기, 시간)
4. gdb 실측 결과
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

# 프로젝트 경로
PROJECT_ROOT = Path("/home/ljy/onnx-mlir")
DEBUG_INFO_DIR = PROJECT_ROOT / "debug_info_investigation"
ARTIFACTS_V3 = DEBUG_INFO_DIR / "artifacts" / "v3"

# 기본 지표: op_type을 소스 파일과 연결하는 규칙
# (ONNX op_type) → (예상 출처 .py 파일명)
OP_TYPE_SOURCE_MAP = {
    "Conv": "conv",
    "Relu": ["activation", "relu"],
    "MaxPool": ["pooling", "pool"],
    "BatchNormalization": ["batchnorm", "norm"],
    "Gemm": ["linear", "fc"],
    "MatMul": ["linear", "matmul"],
    "Add": ["activation", "add"],
    "Mul": ["ops", "mul"],
    "Reshape": ["view", "reshape"],
    "Flatten": ["flatten"],
}

def is_framework_path(filepath: str) -> bool:
    """프레임워크 내부 경로 판정."""
    markers = ["/site-packages/", "/dist-packages/", "/lib/python"]
    return any(m in filepath for m in markers)

def pick_user_frame(frames: list[dict]) -> dict[str, Any] | None:
    """사용자 코드 프레임 선택 (가장 안쪽 사용자 프레임)."""
    if not frames:
        return None
    for fr in reversed(frames):
        if not is_framework_path(fr["file"]):
            return fr
    return frames[-1] if frames else None

def matches_op_type(op_type: str, source_file: str) -> bool:
    """op_type이 선택된 source file과 일치하는지 판정."""
    basename = os.path.basename(source_file).lower()
    stem = os.path.splitext(basename)[0].lower()

    if op_type not in OP_TYPE_SOURCE_MAP:
        return False

    expected = OP_TYPE_SOURCE_MAP[op_type]
    if isinstance(expected, str):
        expected = [expected]

    return any(exp in stem for exp in expected)

def extract_metrics(model_dir: Path) -> dict[str, Any]:
    """모델 디렉토리에서 지표 추출."""
    model_name = model_dir.name

    # 파일 경로
    locmap_path = list(model_dir.glob("*.locmap.json"))[0] if list(model_dir.glob("*.locmap.json")) else None
    pipeline_report_path = list(model_dir.glob("*_pipeline_report.json"))[0] if list(model_dir.glob("*_pipeline_report.json")) else None
    kernels_c = list(model_dir.glob("*_kernels.c"))[0] if list(model_dir.glob("*_kernels.c")) else None
    kernels_line_c = list(model_dir.glob("*_kernels.line.c"))[0] if list(model_dir.glob("*_kernels.line.c")) else None

    result = {"model": model_name}

    # locmap 읽기
    if not locmap_path:
        result["error"] = "locmap.json not found"
        return result

    with open(locmap_path) as f:
        locmap = json.load(f)

    nodes_dict = locmap.get("nodes", {})
    nodes_without = locmap.get("nodes_without_location", [])

    # 기본 지표
    total_nodes = len(nodes_dict) + len(nodes_without)
    nodes_with_location = len(nodes_dict)

    # 매핑 정확도
    matched_count = 0
    unmatched_count = 0
    unmatched_nodes = []

    for node_name, node_info in nodes_dict.items():
        op_type = node_info.get("op_type", "")
        file = node_info.get("file", "")

        if matches_op_type(op_type, file):
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_nodes.append({
                "name": node_name,
                "op_type": op_type,
                "selected_file": os.path.basename(file),
            })

    result["basic_metrics"] = {
        "total_nodes": total_nodes,
        "nodes_with_location": nodes_with_location,
        "nodes_without_location": len(nodes_without),
        "matching_accuracy_default": {
            "matched": matched_count,
            "unmatched": unmatched_count,
            "accuracy": matched_count / nodes_with_location if nodes_with_location else 0,
        }
    }

    result["unmatched_examples"] = unmatched_nodes[:5]  # 처음 5개만

    # pipeline_report에서 inject 통계
    if pipeline_report_path:
        with open(pipeline_report_path) as f:
            report = json.load(f)

        inject_stats = report.get("inject_stats", {})
        result["inject_stats"] = {
            "matched_nodes_in_join": len(inject_stats.get("matched", [])),
            "unknown_nodes": len(inject_stats.get("unknown", [])),
            "directives_injected": inject_stats.get("directives", 0),
        }

        verify = report.get("verify", {})
        result["verification"] = {
            "gcc_debug_section_ok": any(
                c.get("desc", "").startswith("gcc -g") and c.get("ok")
                for c in verify.get("checks", [])
            ),
            "dwarf_py_files_found": any(
                ".debug_line 에 원본" in c.get("desc", "") and c.get("ok")
                for c in verify.get("checks", [])
            ),
            "gdb_breakpoint_ok": any(
                "gdb:" in c.get("desc", "") and c.get("ok")
                for c in verify.get("checks", [])
            ),
        }

    # C 파일 크기/줄 수
    if kernels_c and kernels_line_c:
        with open(kernels_c) as f:
            c_lines = len(f.readlines())
        with open(kernels_line_c) as f:
            line_c_lines = len(f.readlines())

        result["code_overhead"] = {
            "kernels_c_lines": c_lines,
            "kernels_line_c_lines": line_c_lines,
            "line_directives_added": line_c_lines - c_lines,
        }

    return result

def main():
    models = sorted([d for d in ARTIFACTS_V3.iterdir() if d.is_dir()])

    results = {}
    for model_dir in models:
        metrics = extract_metrics(model_dir)
        results[metrics["model"]] = metrics

    # 결과 출력
    print("=" * 80)
    print("PAPER EVALUATION METRICS")
    print("=" * 80)
    print()

    # 표 1: 기본 지표
    print("TABLE 1: Basic Metrics")
    print("-" * 80)
    header = f"{'Model':<25} {'Nodes':<8} {'Located':<8} {'Accuracy':<12} {'Inject OK':<10}"
    print(header)
    print("-" * 80)

    for model_name in sorted(results.keys()):
        m = results[model_name]
        if "error" in m:
            print(f"{model_name:<25} ERROR: {m['error']}")
            continue

        metrics = m.get("basic_metrics", {})
        inject = m.get("inject_stats", {})
        acc = metrics.get("matching_accuracy_default", {}).get("accuracy", 0)
        inject_ok = inject.get("matched_nodes_in_join", 0)

        print(f"{model_name:<25} {metrics['total_nodes']:<8} {metrics['nodes_with_location']:<8} {acc:.2%}         {inject_ok:<10}")

    print()

    # 표 2: 오버헤드
    print("TABLE 2: Code Overhead")
    print("-" * 80)
    header = f"{'Model':<25} {'Base C':<10} {'With #line':<12} {'Added Lines':<12}"
    print(header)
    print("-" * 80)

    for model_name in sorted(results.keys()):
        m = results[model_name]
        if "code_overhead" not in m:
            continue

        oh = m["code_overhead"]
        print(f"{model_name:<25} {oh['kernels_c_lines']:<10} {oh['kernels_line_c_lines']:<12} {oh['line_directives_added']:<12}")

    print()

    # 불일치 사례
    print("MAPPING ACCURACY DETAILS")
    print("-" * 80)

    for model_name in sorted(results.keys()):
        m = results[model_name]
        if "error" in m:
            continue

        metrics = m.get("basic_metrics", {})
        unmatched = metrics.get("matching_accuracy_default", {}).get("unmatched", 0)

        if unmatched > 0:
            print(f"\n{model_name}: {unmatched} unmatched nodes")
            for ex in m.get("unmatched_examples", []):
                print(f"  - {ex['name']}: {ex['op_type']} → {ex['selected_file']}")

    print()

    # 검증 결과
    print("VERIFICATION RESULTS")
    print("-" * 80)

    for model_name in sorted(results.keys()):
        m = results[model_name]
        if "verification" not in m:
            continue

        v = m["verification"]
        print(f"{model_name}:")
        print(f"  gcc -g DWARF section: {'✓' if v['gcc_debug_section_ok'] else '✗'}")
        print(f"  .py files in DWARF:   {'✓' if v['dwarf_py_files_found'] else '✗'}")
        print(f"  gdb breakpoint bind:  {'✓' if v['gdb_breakpoint_ok'] else '✗'}")

    # JSON 출력 (상세)
    print()
    print("=" * 80)
    print("DETAILED JSON")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
