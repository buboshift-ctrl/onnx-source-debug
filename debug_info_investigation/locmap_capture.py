#!/usr/bin/env python3
"""
(v3) 하위호환 shim — 실제 구현은 `info_track.py` 로 이동했다.

정보 트랙(admit 단계): 원본 ONNX 의 node 별 metadata_props 를 읽어 locmap(node.name → 원본
.py 파일:줄) 을 만든다. 자세한 설명·스키마는 info_track.py 참고.

    python3 locmap_capture.py MODEL.onnx -o MODEL.locmap.json      # = info_track.py
"""
from info_track import build_locmap, dumps_deterministic, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
