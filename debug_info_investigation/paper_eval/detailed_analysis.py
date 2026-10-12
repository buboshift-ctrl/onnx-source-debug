#!/usr/bin/env python3
"""
논문용 상세 분석:
1. 여러 프레임 선택 정책 비교
2. Conv-BatchNorm fusion 감지
3. op_type 별 모든 통계
"""
import json
import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/home/ljy/onnx-mlir")
DEBUG_INFO_DIR = PROJECT_ROOT / "debug_info_investigation"
ARTIFACTS_V3 = DEBUG_INFO_DIR / "artifacts" / "v3"

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
    "Clip": ["activation", "clip"],
    "ReduceMean": ["pool", "reduce"],
}

def is_framework_path(filepath: str) -> bool:
    """PyTorch/torchvision 내부 경로 판정."""
    markers = ["/site-packages/", "/dist-packages/", "/lib/python"]
    return any(m in filepath for m in markers)

def is_user_model_path(filepath: str) -> bool:
    """사용자 모델 코드 경로 판정 (torchvision 제외)."""
    if "/site-packages/torchvision" in filepath or "/dist-packages/torchvision" in filepath:
        return False  # torchvision도 프레임워크로 간주
    markers = ["/site-packages/", "/dist-packages/", "/lib/python"]
    return not any(m in filepath for m in markers)

def analyze_node(node: dict, policies: dict) -> dict:
    """한 노드를 여러 정책으로 분석."""
    op_type = node["op_type"]

    # Policy 1: 기본 (가장 안쪽 사용자 프레임)
    frames = node.get("stack_frames", [])

    # Policy 2: op_type과 일치하는 프레임 우선
    def pick_matching_op_type(frames, op_type):
        if not frames:
            return None
        # op_type과 일치하는 프레임을 찾음
        for fr in reversed(frames):
            file = fr["file"]
            if is_framework_path(file):
                continue
            basename = os.path.basename(file).lower()
            expected = OP_TYPE_SOURCE_MAP.get(op_type, [])
            if isinstance(expected, str):
                expected = [expected]
            if any(exp in basename for exp in expected):
                return fr
        # 없으면 기본 정책으로 폴백
        for fr in reversed(frames):
            if not is_framework_path(fr["file"]):
                return fr
        return frames[-1] if frames else None

    # Policy 3: 사용자 프레임 우선
    def pick_user_frame(frames):
        if not frames:
            return None
        for fr in reversed(frames):
            if is_user_model_path(fr["file"]):
                return fr
        # 없으면 프레임워크 아닌 것 찾기
        for fr in reversed(frames):
            if not is_framework_path(fr["file"]):
                return fr
        return frames[-1] if frames else None

    results = {}

    # Policy 1: 기본 (현재 동작)
    picked = None
    for fr in reversed(frames):
        if not is_framework_path(fr["file"]):
            picked = fr
            break
    if not picked:
        picked = frames[-1] if frames else None
    if picked:
        file = os.path.basename(picked["file"])
        matches = matches_op_type(op_type, file)
        results["policy1_basic"] = {
            "file": file,
            "line": picked["line"],
            "matches": matches,
        }

    # Policy 2: op_type 일치 우선
    picked = pick_matching_op_type(frames, op_type)
    if picked:
        file = os.path.basename(picked["file"])
        matches = matches_op_type(op_type, file)
        results["policy2_matching"] = {
            "file": file,
            "line": picked["line"],
            "matches": matches,
        }

    # Policy 3: 사용자 코드 우선
    picked = pick_user_frame(frames)
    if picked:
        file = os.path.basename(picked["file"])
        matches = matches_op_type(op_type, file)
        results["policy3_user"] = {
            "file": file,
            "line": picked["line"],
            "matches": matches,
        }

    return results

def matches_op_type(op_type: str, source_file: str) -> bool:
    """op_type과 source_file이 일치하는가."""
    basename = source_file.lower()
    stem = os.path.splitext(basename)[0].lower()

    if op_type not in OP_TYPE_SOURCE_MAP:
        return False

    expected = OP_TYPE_SOURCE_MAP[op_type]
    if isinstance(expected, str):
        expected = [expected]

    return any(exp in stem for exp in expected)

def detect_fusion(node: dict) -> str:
    """Conv-BN fusion 등 감지."""
    frames = node.get("stack_frames", [])
    filenames = [os.path.basename(f["file"]) for f in frames]

    # Conv인데 batchnorm.py가 있으면 fusion
    if node["op_type"] == "Conv" and "batchnorm.py" in filenames:
        return "Conv-BN fusion"

    # Relu인데 다른 연산의 프레임이 있으면
    if node["op_type"] == "Relu":
        for fname in filenames:
            if "batchnorm" in fname:
                return "BN-Relu fusion"

    return "normal"

def main():
    models = sorted([d for d in ARTIFACTS_V3.iterdir() if d.is_dir()])

    for model_dir in models:
        model_name = model_dir.name
        locmap_path = list(model_dir.glob("*.locmap.json"))[0]

        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        with open(locmap_path) as f:
            locmap = json.load(f)

        nodes_dict = locmap.get("nodes", {})

        # 1. Op_type별 통계
        print("\n### Op_type Distribution ###")
        op_counts = defaultdict(int)
        op_fusion = defaultdict(list)

        for node_name, node in nodes_dict.items():
            op_type = node["op_type"]
            op_counts[op_type] += 1

            fusion = detect_fusion(node)
            if fusion != "normal":
                op_fusion[op_type].append((node_name, fusion))

        for op_type in sorted(op_counts.keys()):
            count = op_counts[op_type]
            fused = len(op_fusion.get(op_type, []))
            if fused > 0:
                print(f"{op_type:20s}: {count:3d} ({fused} fused)")
            else:
                print(f"{op_type:20s}: {count:3d}")

        # 2. Fusion 분석
        if any(op_fusion.values()):
            print("\n### Fusion Detection ###")
            for op_type, instances in sorted(op_fusion.items()):
                print(f"{op_type}:")
                for node_name, fusion in instances[:3]:  # 처음 3개만
                    print(f"  - {node_name}: {fusion}")

        # 3. 정책별 정확도
        print("\n### Frame Selection Policy Comparison ###")

        policy_stats = {"policy1_basic": 0, "policy2_matching": 0, "policy3_user": 0}
        policy_totals = {"policy1_basic": 0, "policy2_matching": 0, "policy3_user": 0}

        for node_name, node in nodes_dict.items():
            results = analyze_node(node, {})
            for policy, res in results.items():
                if policy in policy_stats:
                    policy_totals[policy] += 1
                    if res["matches"]:
                        policy_stats[policy] += 1

        for policy in sorted(policy_totals.keys()):
            total = policy_totals[policy]
            matched = policy_stats[policy]
            if total > 0:
                acc = matched / total
                policy_name = policy.replace("policy", "").replace("_", " ")
                print(f"{policy_name:20s}: {acc:.1%} ({matched}/{total})")

        # 4. 특수한 사례
        print("\n### Notable Cases ###")

        # Conv가 비정상 위치로 매핑된 사례
        conv_issues = []
        for node_name, node in nodes_dict.items():
            if node["op_type"] == "Conv":
                fname = os.path.basename(node["file"]).lower()
                if "conv" not in fname:
                    conv_issues.append((node_name, node["file"]))

        if conv_issues:
            print(f"Conv nodes in unexpected files: {len(conv_issues)}")
            for name, file in conv_issues[:5]:
                print(f"  - {name}: {os.path.basename(file)}")

        # Relu의 위치
        relu_files = set()
        for node_name, node in nodes_dict.items():
            if node["op_type"] == "Relu":
                relu_files.add(os.path.basename(node["file"]))

        if len(relu_files) > 1:
            print(f"\nRelu in multiple files: {relu_files}")
        else:
            print(f"\nRelu file: {relu_files}")

if __name__ == "__main__":
    main()
