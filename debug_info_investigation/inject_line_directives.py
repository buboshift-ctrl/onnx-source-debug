#!/usr/bin/env python3
"""
(v3) 하위호환 shim — 실제 구현이 나뉘어 이동했다:

  * 조인 로직(`#line` 투영)          → `join.py`         (`inject(c_source, locmap) -> str`)
  * 데모 C 생성기                     → `code_track.py`   (`get_generated_c(onnx_path) -> str`)
  * 데모+검증 오케스트레이션          → `run_full_pipeline.py`

이 파일은 기존 명령줄 사용법을 계속 지원하기 위한 얇은 어댑터다.

    # 일반 모드 (임의의 생성 C 에 투영) — join.py 로 위임
    python3 inject_line_directives.py kernels.c MODEL.locmap.json -o kernels.line.c

    # 데모 모드 — run_full_pipeline.py 로 위임 (locmap.json 또는 .onnx 둘 다 받음)
    python3 inject_line_directives.py --demo MODEL.locmap.json --workdir ./artifacts/partB
"""
from __future__ import annotations

import json
import os
import sys


def _demo(argv: list[str]) -> int:
    import argparse

    import run_full_pipeline

    ap = argparse.ArgumentParser(prog="inject_line_directives.py --demo")
    ap.add_argument("target", help="locmap JSON 또는 .onnx 파일")
    ap.add_argument("--workdir", default="./partB_out")
    ap.add_argument("--relative-to", default=None)
    args = ap.parse_args(argv)

    target = args.target
    if target.endswith(".onnx"):
        onnx_path = target
    else:
        with open(target, encoding="utf-8") as fh:
            meta = json.load(fh).get("_meta", {})
        onnx_name = meta.get("onnx_file", "")
        # locmap 과 같은 디렉터리에서 onnx 를 찾는다
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(target)), onnx_name)
        if not os.path.isfile(onnx_path):
            print(f"locmap 의 onnx_file 을 못 찾음: {onnx_path}", file=sys.stderr)
            return 2

    rel = args.relative_to or run_full_pipeline.ROOT
    report = run_full_pipeline.run(onnx_path, args.workdir, rel)
    if "verify" not in report:
        print(report.get("error", "실패") + "\n" + report.get("stderr", ""), file=sys.stderr)
        return 1
    verdict = report["verify"]
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("PART B: PASS" if verdict["ok"] else "PART B: FAIL", file=sys.stderr)
    return 0 if verdict["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--demo" in argv:
        argv.remove("--demo")
        return _demo(argv)
    import join
    return join.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
