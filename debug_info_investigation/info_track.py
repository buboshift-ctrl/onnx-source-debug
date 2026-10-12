#!/usr/bin/env python3
"""
정보 트랙 (information track) — admit 단계.

역할: 원본 ONNX 파일의 node 별 `metadata_props`(특히 `pkg.torch.onnx.stack_trace`)를 읽어,
`node.name` → 원본 `.py` 파일:줄 매핑(locmap)을 만든다. onnx-mlir 임포터가 이 정보를 버리기
때문에(근거: debug_info_investigation.md 2순위), admit 단계에서 onnx-mlir 에 넘기기 *전에*
따로 떠 둔다.

아키텍처상의 위치 (TASK_formalize_architecture_v3.md):
  - 이 모듈은 **코드 트랙(code_track.py)의 존재를 몰라도 된다.** 완전히 독립.
  - 두 트랙은 `join.py` 에서 딱 한 번 만나며, 조인 키는 오직 `node.name` (== onnx-mlir `onnx_node_name`).

공개 API:
  build_locmap(onnx_path: str, *, relative_to: str | None = None) -> dict
  dumps_deterministic(locmap: dict) -> str

결정성: 타임스탬프/난수 없음. 같은 .onnx → 바이트 동일 JSON
(노드는 그래프 순서 유지, ensure_ascii=False, 마지막 개행 1개).

CLI (하위호환: `locmap_capture.py` 가 이 main 을 그대로 부른다):
    python3 info_track.py MODEL.onnx -o MODEL.locmap.json
    python3 info_track.py MODEL.onnx --relative-to /workdir/onnx-mlir --stdout
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from typing import Any

SCHEMA_VERSION = 1

# locmap `_meta.tool` 필드값. v3 에서 파일은 info_track.py 로 이동했지만,
# locmap 스키마/산출물 바이트 안정성을 위해 값은 유지한다.
_TOOL_ID = "locmap_capture.py"

# "File "<path>", line <n>, in <func>\n    <source>"  (표준 파이썬 트레이스백 프레임)
_FRAME_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>[^\n]+)'
    r'(?:\n[ \t]+(?P<source>[^\n]*))?'
)

# 사용자 코드가 아니라 "프레임워크 내부"로 볼 경로 조각
_FRAMEWORK_MARKERS = ("/site-packages/", "/dist-packages/", "/lib/python")


def _parse_literal_list(value: str) -> list[str]:
    """"['a', 'b']" 형태의 파이썬 리터럴 문자열을 리스트로. 실패하면 원문을 1-요소 리스트로."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [value]


def _parse_stack_trace(trace: str) -> list[dict[str, Any]]:
    """`pkg.torch.onnx.stack_trace` 문자열 → 프레임 리스트(위에서 아래, torch 는 사용자 프레임을 먼저 놓는다)."""
    frames: list[dict[str, Any]] = []
    for m in _FRAME_RE.finditer(trace):
        frames.append(
            {
                "file": m.group("file"),
                "line": int(m.group("line")),
                "func": m.group("func").strip(),
                "source": (m.group("source") or "").strip(),
            }
        )
    return frames


def _pick_user_frame(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    """중단점을 걸 프레임: 프레임워크 내부가 아닌 첫 프레임. 없으면 첫 프레임."""
    if not frames:
        return None
    for fr in frames:
        if not any(marker in fr["file"] for marker in _FRAMEWORK_MARKERS):
            return fr
    return frames[0]


def _module_scope_from_name_scopes(name_scopes: list[str]) -> str:
    """['', 'fc1', 'linear'] → 'fc1.linear'."""
    return ".".join(s for s in name_scopes if s)


def _rel(path: str, root: str | None) -> str:
    """root 아래에 있는 경로만 상대경로로. 밖으로 벗어나면(예: torch dist-packages) 절대경로 유지."""
    if not root:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith(".." + os.sep) or rel == "..":
        return path
    return rel


def _metadata_to_dict(node) -> dict[str, str]:
    return {kv.key: kv.value for kv in node.metadata_props}


def _build_from_model(model, *, onnx_path: str, relative_to: str | None) -> dict[str, Any]:
    """
    locmap 스키마 (schema_version=1):

    {
      "_meta": {
        "schema_version": 1,
        "tool": "locmap_capture.py",
        "onnx_file": "<basename>",
        "onnx_sha256": "<hex>",
        "producer": "pytorch 2.14.0+cpu",
        "ir_version": 10,
        "opset": [["", 20]],
        "node_count": <int>,
        "nodes_with_location": <int>,
        "join_key": "onnx_node_name",
        "path_base": "<relative_to 또는 null>"
      },
      "nodes": {
        "<node.name == onnx_node_name>": {
          "op_type": "Gemm",
          "file":   "<원본 .py 파일 경로>",          # 중단점 대상
          "line":   23,                               # 중단점 대상
          "source": "x = self.relu(self.fc1(x))",     # 참고용 소스 한 줄
          "func":   "forward",
          "module_scope":    "fc1.linear",            # name_scopes 유래(디버거 트리뷰용)
          "namespace":       ": __main__.TinyCNN/fc1: ...",   # torch 원문
          "class_hierarchy": ["__main__.TinyCNN", "...Linear", "aten.linear.default"],
          "fx_node":         "%linear : ... = call_function[...]",
          "stack_frames":    [ {file,line,func,source}, ... ]   # 전체 콜스택(사용자+torch)
        },
        ...
      },
      "nodes_without_location": ["<node.name>", ...]   # metadata 없어서 매핑 못 뜬 노드
    }
    """
    with open(onnx_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()

    nodes: dict[str, Any] = {}
    missing: list[str] = []

    for idx, node in enumerate(model.graph.node):
        name = node.name or f"__anon_{idx}__{node.op_type}"
        md = _metadata_to_dict(node)

        entry: dict[str, Any] = {"op_type": node.op_type}

        stack_trace = md.get("pkg.torch.onnx.stack_trace", "")
        frames = _parse_stack_trace(stack_trace) if stack_trace else []
        user_frame = _pick_user_frame(frames)

        if user_frame is not None:
            entry["file"] = _rel(user_frame["file"], relative_to)
            entry["line"] = user_frame["line"]
            entry["source"] = user_frame["source"]
            entry["func"] = user_frame["func"]
        else:
            missing.append(name)

        name_scopes = _parse_literal_list(md.get("pkg.torch.onnx.name_scopes", "[]"))
        module_scope = _module_scope_from_name_scopes(name_scopes)
        if module_scope:
            entry["module_scope"] = module_scope
        if "namespace" in md:
            entry["namespace"] = md["namespace"]
        if "pkg.torch.onnx.class_hierarchy" in md:
            entry["class_hierarchy"] = _parse_literal_list(
                md["pkg.torch.onnx.class_hierarchy"]
            )
        if "pkg.torch.onnx.fx_node" in md:
            entry["fx_node"] = md["pkg.torch.onnx.fx_node"]
        if frames:
            entry["stack_frames"] = [
                {**fr, "file": _rel(fr["file"], relative_to)} for fr in frames
            ]

        nodes[name] = entry

    opset = [[o.domain, o.version] for o in model.opset_import]

    return {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "tool": _TOOL_ID,
            "onnx_file": os.path.basename(onnx_path),
            "onnx_sha256": sha,
            "producer": f"{model.producer_name} {model.producer_version}".strip(),
            "ir_version": model.ir_version,
            "opset": opset,
            "node_count": len(model.graph.node),
            "nodes_with_location": len(model.graph.node) - len(missing),
            "join_key": "onnx_node_name",
            "path_base": relative_to,
        },
        "nodes": nodes,
        "nodes_without_location": missing,
    }


def build_locmap(onnx_path: str, *, relative_to: str | None = None) -> dict[str, Any]:
    """ONNX 파일을 로드해 locmap dict 를 만든다. 코드 트랙과 무관하게 독립 동작."""
    import onnx  # 지연 import: onnx 없이 모듈을 import 만 하는 경우를 위해

    model = onnx.load(onnx_path)
    return _build_from_model(model, onnx_path=onnx_path, relative_to=relative_to)


def dumps_deterministic(locmap: dict[str, Any]) -> str:
    """항상 동일 바이트열. (딕셔너리 삽입 순서 = 그래프 순서, sort_keys 안 씀 — 순서 자체가 정보)"""
    return json.dumps(locmap, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="정보 트랙 — ONNX locmap 캡처 (admit 단계)")
    ap.add_argument("onnx", help="dynamo=True 로 export 한 .onnx 파일")
    ap.add_argument("-o", "--output", help="출력 JSON 경로 (기본: <onnx>.locmap.json)")
    ap.add_argument("--stdout", action="store_true", help="파일 대신 표준출력으로")
    ap.add_argument(
        "--relative-to",
        metavar="DIR",
        help="locmap 안의 파일 경로를 이 디렉터리 기준 상대경로로 기록",
    )
    args = ap.parse_args(argv)

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("onnx 모듈이 필요합니다: pip install onnx", file=sys.stderr)
        return 2

    if not os.path.isfile(args.onnx):
        print(f"파일 없음: {args.onnx}", file=sys.stderr)
        return 2

    locmap = build_locmap(args.onnx, relative_to=args.relative_to)
    text = dumps_deterministic(locmap)

    meta = locmap["_meta"]
    if meta["nodes_with_location"] == 0:
        print(
            "경고: 위치 정보를 뜬 노드가 0개입니다. dynamo=True 로 export 한 파일이 맞나요? "
            "(legacy exporter 는 metadata_props 를 안 남깁니다)",
            file=sys.stderr,
        )

    if args.stdout:
        sys.stdout.write(text)
    else:
        out = args.output or (args.onnx + ".locmap.json")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(
            f"locmap 기록: {out}  "
            f"({meta['nodes_with_location']}/{meta['node_count']} 노드 위치 매핑)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
