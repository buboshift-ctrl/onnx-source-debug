"""
"admit / mlir-arx(PNU) / tail" 3단계 파이프라인을 세로 사이드바 형태로 그리는 SVG 생성기.

CLAUDE.md 원문 구조를 그대로 시각화:
    ONNX 모델 → (admit 단계 - 우리가 만짐) → mlir-arx(PNU, 못 건드림) → (tail 단계 - 우리 소유)
    → host-mode 실행 아티팩트

PPT에 복사-붙여넣기 할 용도라서 PPT는 건드리지 않고 SVG 파일만 별도로 생성한다.
SVG를 쓰는 이유: 이 컨테이너엔 한글 폰트가 없어서 PNG로 만들면 한글이 깨진다(네모 박스로 나옴).
SVG는 텍스트를 문자 그대로 담고 있어서, 실제로 여는/붙여넣는 쪽(사용자 PC의 PowerPoint)의
폰트로 렌더링되므로 한글이 정상적으로 보인다.

산출: pipeline_sidebar_overview.svg / _admit.svg / _mlir_arx.svg / _tail.svg
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 팔레트 (report_assignment.pptx/build_slides.py 와 동일 계열) --------------
NAVY = "#1B2A4A"
ACCENT = "#2E6F95"      # admit/tail 공통 "우리 소유" 색
LOCKED = "#8A6D3B"       # mlir-arx "못 건드림" 강조색(호박색 계열 - 경고 느낌)
LOCKED_MUTED_HATCH = "#8A6D3B"
GRAY_TEXT = "#5A5F66"
ENDPOINT_FILL = "#EEF1F4"
ENDPOINT_BORDER = "#C7CCD1"
ENDPOINT_TEXT = "#5A5F66"
MUTED_FILL = "#E9ECEF"
MUTED_BORDER = "#C7CCD1"
MUTED_TEXT = "#9AA0A6"
WHITE = "#FFFFFF"
BG = "#FFFFFF"

FONT = "'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif"

W = 340
BOX_W = 300
BOX_X = (W - BOX_W) / 2
TOP_MARGIN = 70
BOX_H_MAIN = 118       # admit / mlir-arx / tail 박스 높이
BOX_H_END = 70         # ONNX모델 / 아티팩트 박스 높이 (끝점, 더 작게)
GAP = 46               # 박스 사이 화살표 구간 높이

STAGES = [
    {"id": "onnx", "kind": "endpoint", "h": BOX_H_END,
     "title": "ONNX 모델", "sub": "입력"},
    {"id": "admit", "kind": "ours", "h": BOX_H_MAIN,
     "title": "admit 단계", "sub": "ONNX 로드 직후", "tag": "우리가 만짐"},
    {"id": "mlirarx", "kind": "locked", "h": BOX_H_MAIN,
     "title": "mlir-arx (PNU)", "sub": "onnx → harx → EmitC", "tag": "절대 안 건드림"},
    {"id": "tail", "kind": "ours", "h": BOX_H_MAIN,
     "title": "tail 단계", "sub": "EmitC → C 생성 이후", "tag": "우리 소유"},
    {"id": "artifact", "kind": "endpoint", "h": BOX_H_END,
     "title": "host-mode 실행 아티팩트", "sub": "출력"},
]


def _wrap(text: str, max_chars: int) -> list[str]:
    """대략적인 글자 수 기준 줄바꿈 (한글은 폭이 넓어서 넉넉하게 잡음)."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _box_style(stage: dict, highlight: str | None) -> dict:
    kind = stage["kind"]
    is_current = (highlight is not None and stage["id"] == highlight)

    if kind == "endpoint":
        return dict(fill=ENDPOINT_FILL, border=ENDPOINT_BORDER, text=ENDPOINT_TEXT,
                     sub_text=ENDPOINT_TEXT, bw=1.5, tag_bg=None, tag_text=None,
                     hatch=False, current=False)

    if is_current:
        main = ACCENT if kind == "ours" else LOCKED
        return dict(fill=main, border=NAVY, text=WHITE, sub_text="#E9F0F5",
                     bw=3, tag_bg="rgba(255,255,255,0.22)", tag_text=WHITE,
                     hatch=(kind == "locked"), current=True)

    # 강조 대상이 아닌 경우 (overview 에서는 highlight=None -> kind별 제 색으로, 옅게)
    if highlight is None:
        main = ACCENT if kind == "ours" else LOCKED
        return dict(fill=main, border=NAVY, text=WHITE, sub_text="#E9F0F5",
                     bw=2, tag_bg="rgba(255,255,255,0.2)", tag_text=WHITE,
                     hatch=(kind == "locked"), current=False)

    return dict(fill=MUTED_FILL, border=MUTED_BORDER, text=MUTED_TEXT,
                 sub_text=MUTED_TEXT, bw=1.5, tag_bg=None, tag_text=None,
                 hatch=False, current=False)


def render(highlight: str | None, out_name: str, page_label: str) -> str:
    y = TOP_MARGIN
    parts: list[str] = []
    centers: list[tuple[float, float]] = []  # (bottom_y_of_box_i, top_y_of_box_i+1) 용

    defs = f"""
  <defs>
    <pattern id="hatch" width="8" height="8" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="8" height="8" fill="transparent"/>
      <line x1="0" y1="0" x2="0" y2="8" stroke="#ffffff" stroke-opacity="0.35" stroke-width="2"/>
    </pattern>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{GRAY_TEXT}"/>
    </marker>
  </defs>
"""
    parts.append(defs)

    box_tops = []
    for i, stage in enumerate(STAGES):
        h = stage["h"]
        box_tops.append((y, h))
        y += h
        if i < len(STAGES) - 1:
            y += GAP

    total_h = y + 50

    parts.append(f'<rect x="0" y="0" width="{W}" height="{total_h}" fill="{BG}"/>')
    parts.append(
        f'<text x="{W/2}" y="30" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13" font-weight="700" letter-spacing="1.5" fill="{GRAY_TEXT}">'
        f'PIPELINE · {page_label}</text>'
    )
    parts.append(f'<rect x="{W/2-18}" y="40" width="36" height="3" fill="{ACCENT}"/>')

    for i, (stage, (top, h)) in enumerate(zip(STAGES, box_tops)):
        style = _box_style(stage, highlight)
        rx = 12
        bx, by, bw_, bh_ = BOX_X, top, BOX_W, h

        # 화살표 (이전 박스 -> 이 박스)
        if i > 0:
            prev_bottom = box_tops[i - 1][0] + box_tops[i - 1][1]
            parts.append(
                f'<line x1="{W/2}" y1="{prev_bottom+4}" x2="{W/2}" y2="{by-6}" '
                f'stroke="{GRAY_TEXT}" stroke-width="2" marker-end="url(#arrow)"/>'
            )

        # "현재 단계" 표시용 좌측 삼각 마커
        if style["current"]:
            parts.append(
                f'<polygon points="{bx-16},{by+bh_/2-10} {bx-16},{by+bh_/2+10} {bx-2},{by+bh_/2}" '
                f'fill="{NAVY}"/>'
            )

        parts.append(
            f'<rect x="{bx}" y="{by}" width="{bw_}" height="{bh_}" rx="{rx}" '
            f'fill="{style["fill"]}" stroke="{style["border"]}" stroke-width="{style["bw"]}"/>'
        )
        if style["hatch"]:
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{bw_}" height="{bh_}" rx="{rx}" fill="url(#hatch)"/>'
            )

        cx = bx + bw_ / 2
        if stage["kind"] == "endpoint":
            parts.append(
                f'<text x="{cx}" y="{by+bh_/2-2}" text-anchor="middle" font-family="{FONT}" '
                f'font-size="15" font-weight="700" fill="{style["text"]}">{stage["title"]}</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{by+bh_/2+18}" text-anchor="middle" font-family="{FONT}" '
                f'font-size="12" fill="{style["sub_text"]}">{stage["sub"]}</text>'
            )
        else:
            ty = by + 34
            parts.append(
                f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="{FONT}" '
                f'font-size="19" font-weight="700" fill="{style["text"]}">{stage["title"]}</text>'
            )
            for j, line in enumerate(_wrap(stage["sub"], 22)):
                parts.append(
                    f'<text x="{cx}" y="{ty+24+j*18}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="13" fill="{style["sub_text"]}">{line}</text>'
                )
            if style["tag_bg"]:
                tag = stage["tag"]
                tag_w = 26 + len(tag) * 9.5
                tx = cx - tag_w / 2
                tag_y = by + bh_ - 30
                parts.append(
                    f'<rect x="{tx}" y="{tag_y}" width="{tag_w}" height="22" rx="11" '
                    f'fill="{style["tag_bg"]}"/>'
                )
                parts.append(
                    f'<text x="{cx}" y="{tag_y+15}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="11" font-weight="600" fill="{style["tag_text"]}">{tag}</text>'
                )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
        f'viewBox="0 0 {W} {total_h}">\n' + "\n".join(parts) + "\n</svg>\n"
    )

    out_path = os.path.join(HERE, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print("생성:", out_path, f"({W}x{int(total_h)})")
    return out_path


if __name__ == "__main__":
    render(None, "pipeline_sidebar_overview.svg", "전체 개요")
    render("admit", "pipeline_sidebar_admit.svg", "admit 단계")
    render("mlirarx", "pipeline_sidebar_mlir_arx.svg", "mlir-arx (건드리지 않음)")
    render("tail", "pipeline_sidebar_tail.svg", "tail 단계")
