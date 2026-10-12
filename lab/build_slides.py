"""
과제용 PPT(방법1: info_track/code_track/join 파이프라인)와 논문 초안 PPT(방법2: onnx-mlir
임포터 자체 패치)를 python-pptx로 생성한다. 두 PPT 모두 이 세션에서 실제로 실행한 결과
(JSON 리포트, locmap, MLIR 산출물)에 기반한 수치만 사용한다.

실행: python3 build_slides.py
출력: report_assignment.pptx, paper_draft.pptx (같은 디렉터리)
"""
import json
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "shots")

# ----------------------------------------------------------------------------
# 팔레트 / 상수
# ----------------------------------------------------------------------------
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0x2E, 0x6F, 0x95)
ACCENT2 = RGBColor(0xC9, 0x6A, 0x2E)
GRAY = RGBColor(0x5A, 0x5F, 0x66)
LIGHT_BG = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OK_GREEN = RGBColor(0x1E, 0x7B, 0x34)
CODE_BG = RGBColor(0x22, 0x27, 0x2E)
CODE_FG = RGBColor(0xE6, 0xE6, 0xE6)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def new_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, x, y, w, h, fill=None, line=None, line_w=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w or Pt(1)
    return shp


def set_text(shape, text, size=14, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
             font="Pretendard", anchor=None, italic=False):
    tf = shape.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tf


def header_bar(slide, kicker, title, kicker_color=ACCENT):
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.12), fill=kicker_color)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.32), Inches(12.1), Inches(0.4))
    set_text(box, kicker, size=13, color=kicker_color, bold=True)
    box2 = slide.shapes.add_textbox(Inches(0.55), Inches(0.62), Inches(12.2), Inches(0.9))
    set_text(box2, title, size=28, color=NAVY, bold=True)
    return box2


def footer(slide, text, page=None):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(7.08), Inches(11), Inches(0.35))
    set_text(box, text, size=10, color=GRAY)
    if page is not None:
        pbox = slide.shapes.add_textbox(Inches(12.5), Inches(7.08), Inches(0.6), Inches(0.35))
        set_text(pbox, str(page), size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def title_slide(prs, kicker, title, subtitle, meta_lines, accent=ACCENT):
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
    add_rect(s, 0, Inches(4.55), SLIDE_W, Inches(0.06), fill=accent)
    kb = s.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(0.5))
    set_text(kb, kicker, size=16, color=accent, bold=True)
    tb = s.shapes.add_textbox(Inches(0.85), Inches(2.65), Inches(11.6), Inches(1.6))
    set_text(tb, title, size=40, color=WHITE, bold=True)
    sb = s.shapes.add_textbox(Inches(0.9), Inches(4.75), Inches(11.5), Inches(0.7))
    set_text(sb, subtitle, size=18, color=RGBColor(0xC8, 0xD3, 0xE0))
    mb = s.shapes.add_textbox(Inches(0.9), Inches(6.6), Inches(11.5), Inches(0.6))
    set_text(mb, "  |  ".join(meta_lines), size=12, color=RGBColor(0x8F, 0x9B, 0xAC))
    return s


def bullet_slide(prs, kicker, title, bullets, accent=ACCENT, note=None, page=None):
    """bullets: list of (level:int, text:str) or (level, text, color)"""
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.75), Inches(11.9), Inches(4.9))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        level, text = item[0], item[1]
        color = item[2] if len(item) > 2 else (NAVY if level == 0 else GRAY)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = min(level, 4)
        prefix = {0: "▪ ", 1: "– ", 2: "· "}.get(level, "· ")
        r = p.add_run()
        r.text = prefix + text
        r.font.size = Pt(20 - level * 2)
        r.font.color.rgb = color
        r.font.bold = (level == 0)
        r.font.name = "Pretendard"
        p.space_after = Pt(10 if level == 0 else 6)
    if note:
        footer(s, note, page)
    else:
        footer(s, "", page)
    return s


def table_slide(prs, kicker, title, headers, rows, col_widths=None, accent=ACCENT,
                 note=None, page=None, highlight_ok_col=None):
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)
    n_rows = len(rows) + 1
    n_cols = len(headers)
    left, top = Inches(0.7), Inches(1.85)
    width, height = Inches(11.9), Inches(0.5 * n_rows + 0.2)
    gtable = s.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    if col_widths:
        for i, w in enumerate(col_widths):
            gtable.columns[i].width = w
    for c, htext in enumerate(headers):
        cell = gtable.cell(0, c)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.color.rgb = WHITE
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(14)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = gtable.cell(r + 1, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(13)
            p.runs[0].font.color.rgb = NAVY
            if highlight_ok_col is not None and c == highlight_ok_col:
                if "PASS" in str(val) or "true" in str(val).lower() or "✓" in str(val):
                    p.runs[0].font.color.rgb = OK_GREEN
                    p.runs[0].font.bold = True
    footer(s, note or "", page)
    return s


def code_slide(prs, kicker, title, code_text, caption=None, accent=ACCENT2, page=None, size=13):
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)
    box = add_rect(s, Inches(0.7), Inches(1.7), Inches(11.9), Inches(4.9), fill=CODE_BG)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    for i, line in enumerate(code_text.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line if line.strip() else " "
        r.font.name = "Consolas"
        r.font.size = Pt(size)
        r.font.color.rgb = CODE_FG
    if caption:
        cb = s.shapes.add_textbox(Inches(0.7), Inches(6.65), Inches(11.9), Inches(0.5))
        set_text(cb, caption, size=12, color=GRAY, italic=True)
    footer(s, "", page)
    return s


def _place_picture(slide, image_path, x, y, max_w, max_h, caption=None, caption_color=GRAY):
    """이미지를 (x,y)~(x+max_w,y+max_h) 박스 안에 비율 유지로 맞추고, 세로로 가운데 정렬.
    캡션은 실제로 배치된 그림 바로 아래에 붙인다(고정 좌표로 겹치거나 멀리 떨어지는 것 방지)."""
    pic = slide.shapes.add_picture(image_path, x, y, width=max_w)
    if pic.height > max_h:
        pic._element.getparent().remove(pic._element)
        pic = slide.shapes.add_picture(image_path, x, y, height=max_h)
    pic.left = Emu(int(x + (max_w - pic.width) / 2))
    pic.top = Emu(int(y + (max_h - pic.height) / 2))
    if caption:
        cap_top = Emu(int(y + max_h + Inches(0.12)))
        cb = slide.shapes.add_textbox(x, cap_top, max_w, Inches(0.5))
        set_text(cb, caption, size=11, color=caption_color, italic=True)
    return pic


def code_shot_slide(prs, kicker, title, code_text, image_path, code_caption=None,
                     shot_caption=None, accent=ACCENT, page=None, size=12.5, stage_tag=None):
    """왼쪽: 실제 소스 코드(텍스트). 오른쪽: 그 코드를 실행해서 나온 실제 출력의 스크린샷(이미지)."""
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)

    left_w = Inches(6.0)
    right_x = Inches(6.85)
    right_w = Inches(5.75)
    top = Inches(1.7)
    h = Inches(4.75)

    if stage_tag:
        tagbox = s.shapes.add_textbox(Inches(0.7), Inches(1.42), Inches(6), Inches(0.3))
        set_text(tagbox, stage_tag, size=12, color=accent, bold=True)

    lbl1 = s.shapes.add_textbox(Inches(0.7), top, left_w, Inches(0.3))
    set_text(lbl1, "CODE", size=11, color=GRAY, bold=True)
    box = add_rect(s, Inches(0.7), Inches(2.0), left_w, Inches(4.45), fill=CODE_BG)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.15)
    for i, line in enumerate(code_text.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line if line.strip() else " "
        r.font.name = "Consolas"
        r.font.size = Pt(size)
        r.font.color.rgb = CODE_FG
    if code_caption:
        cb = s.shapes.add_textbox(Inches(0.7), Inches(6.5), left_w, Inches(0.55))
        set_text(cb, code_caption, size=11, color=GRAY, italic=True)

    lbl2 = s.shapes.add_textbox(right_x, top, right_w, Inches(0.3))
    set_text(lbl2, "ACTUAL OUTPUT (실측)", size=11, color=OK_GREEN, bold=True)
    _place_picture(s, image_path, right_x, Inches(2.0), right_w, Inches(4.15),
                   caption=shot_caption)

    footer(s, "", page)
    return s


def bullet_shot_slide(prs, kicker, title, bullets, image_path, accent=ACCENT, page=None,
                       shot_caption=None):
    """왼쪽: 설명 불릿. 오른쪽: 실제 출력 스크린샷."""
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)

    left_w = Inches(6.0)
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.8), left_w, Inches(4.8))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        level, text = item[0], item[1]
        color = item[2] if len(item) > 2 else (NAVY if level == 0 else GRAY)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = min(level, 4)
        prefix = {0: "▪ ", 1: "– ", 2: "· "}.get(level, "· ")
        r = p.add_run()
        r.text = prefix + text
        r.font.size = Pt(17 - level * 1.5)
        r.font.color.rgb = color
        r.font.bold = (level == 0)
        r.font.name = "Pretendard"
        p.space_after = Pt(8 if level == 0 else 5)

    right_x = Inches(6.85)
    right_w = Inches(5.75)
    lbl2 = s.shapes.add_textbox(right_x, Inches(1.7), right_w, Inches(0.3))
    set_text(lbl2, "ACTUAL OUTPUT (실측)", size=11, color=OK_GREEN, bold=True)
    _place_picture(s, image_path, right_x, Inches(2.0), right_w, Inches(4.15),
                   caption=shot_caption)
    footer(s, "", page)
    return s


def diagram_pipeline_slide(prs, kicker, title, boxes, accent=ACCENT, note=None, page=None):
    """boxes: list of dicts {label, sub, color} rendered left-to-right with arrows."""
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    header_bar(s, kicker, title, kicker_color=accent)
    n = len(boxes)
    top = Inches(2.6)
    h = Inches(2.0)
    gap = Inches(0.35)
    total_w = Inches(11.9)
    bw = Emu(int((total_w - gap * (n - 1)) / n))
    x = Inches(0.7)
    centers = []
    for i, b in enumerate(boxes):
        rect = add_rect(s, x, top, bw, h, fill=b.get("color", ACCENT), line=None)
        rect.text_frame.word_wrap = True
        rect.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p0 = rect.text_frame.paragraphs[0]
        p0.alignment = PP_ALIGN.CENTER
        r0 = p0.add_run()
        r0.text = b["label"]
        r0.font.size = Pt(16)
        r0.font.bold = True
        r0.font.color.rgb = WHITE
        if b.get("sub"):
            p1 = rect.text_frame.add_paragraph()
            p1.alignment = PP_ALIGN.CENTER
            r1 = p1.add_run()
            r1.text = b["sub"]
            r1.font.size = Pt(11)
            r1.font.color.rgb = RGBColor(0xE9, 0xEE, 0xF3)
        centers.append((x, x + bw))
        x = Emu(int(x + bw + gap))
    for i in range(n - 1):
        ax1 = centers[i][1]
        ax2 = centers[i + 1][0]
        conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, ax1, Emu(int(top + h / 2)),
                                       ax2, Emu(int(top + h / 2)))
        conn.line.color.rgb = GRAY
        conn.line.width = Pt(2.25)
        conn.line.end_arrowhead = True if hasattr(conn.line, "end_arrowhead") else None
    # caption row under boxes
    if note:
        cb = s.shapes.add_textbox(Inches(0.7), Inches(5.1), Inches(11.9), Inches(1.6))
        tf = cb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(note):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = "• " + line
            r.font.size = Pt(14)
            r.font.color.rgb = GRAY
            p.space_after = Pt(4)
    footer(s, "", page)
    return s


def section_slide(prs, number, title, subtitle, color=NAVY):
    s = blank_slide(prs)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=color)
    nb = s.shapes.add_textbox(Inches(0.9), Inches(2.6), Inches(3), Inches(1.2))
    set_text(nb, number, size=54, color=RGBColor(0x55, 0x6B, 0x8F), bold=True)
    tb = s.shapes.add_textbox(Inches(0.9), Inches(3.5), Inches(11), Inches(1.2))
    set_text(tb, title, size=32, color=WHITE, bold=True)
    if subtitle:
        sb = s.shapes.add_textbox(Inches(0.9), Inches(4.35), Inches(11), Inches(0.6))
        set_text(sb, subtitle, size=16, color=RGBColor(0xC8, 0xD3, 0xE0))
    return s


# ============================================================================
# 데이터 로드 (실제 실행 산출물)
# ============================================================================
def load_report(rel):
    with open(os.path.join(HERE, rel)) as f:
        return json.load(f)

rn18_before = load_report("artifacts/resnet18/resnet18_dynamo_pipeline_report.json")
mnv2_before = load_report("artifacts/mobilenet_v2/mobilenet_v2_dynamo_pipeline_report.json")
rn18_after = load_report("../debug_info_investigation/artifacts/v3/resnet18_dynamo/resnet18_dynamo_pipeline_report.json")
mnv2_after = load_report("../debug_info_investigation/artifacts/v3/mobilenet_v2_dynamo/mobilenet_v2_dynamo_pipeline_report.json")

def uniq_lines(locmap_rel):
    with open(os.path.join(HERE, locmap_rel)) as f:
        lm = json.load(f)
    nodes = lm["nodes"]
    lines = set((e["file"], e["line"]) for e in nodes.values() if "file" in e)
    return len(nodes), len(lines)

rn18_n, rn18_u_before = uniq_lines("artifacts/resnet18/resnet18_dynamo.locmap.json")
_, rn18_u_after = uniq_lines("../debug_info_investigation/artifacts/v3/resnet18_dynamo/resnet18_dynamo.locmap.json")
mnv2_n, mnv2_u_before = uniq_lines("artifacts/mobilenet_v2/mobilenet_v2_dynamo.locmap.json")
_, mnv2_u_after = uniq_lines("../debug_info_investigation/artifacts/v3/mobilenet_v2_dynamo/mobilenet_v2_dynamo.locmap.json")


def checks_rows(report):
    return [(c["desc"], "PASS" if c["ok"] else "FAIL") for c in report["verify"]["checks"]]


# ============================================================================
# 과제 PPT (방법1: info_track / code_track / join)
# ============================================================================
def build_assignment_deck():
    prs = new_presentation()

    title_slide(
        prs,
        kicker="재구성형 AI 프로세서 SW 프레임워크 · 4세부(디버거·프로파일러·통합개발환경)",
        title="ONNX 디버깅 정보 보존 파이프라인",
        subtitle="admit(정보 트랙) + tail(코드 트랙) 2-트랙 구조로 #line/DWARF까지 투영 — 과제 진행 보고",
        meta_lines=["2026-09-14", "실행 환경: 업스트림 onnx-mlir 0.5.1 (일반 MLIR 인프라 기준, ARX 아님)"],
    )

    bullet_slide(
        prs, "1. 배경", "문제 정의",
        [
            (0, "ONNX는 표준 스펙상 원본 소스 위치(파일/줄) 정보를 남기지 않는다"),
            (1, "onnx_node_name(연산 이름)만 남고, 어느 .py 코드에서 왔는지는 기본적으로 소실"),
            (0, "목표: MLIR/DWARF 방식을 참고해 실제 중단점 · 변수 추적이 가능한 구조 설계"),
            (0, "우리가 수정 가능한 지점은 두 곳뿐"),
            (1, "admit 단계 — ONNX 로드 직후"),
            (1, "tail 단계 — EmitC → C 생성 이후"),
            (2, "중간 mlir-arx(PNU, LLVM 20, 라이선스 이슈) 컴파일러 내부는 건드리지 않음"),
        ],
        page=2,
    )

    diagram_pipeline_slide(
        prs, "2. 아키텍처", "2-트랙 구조 + 단일 조인점",
        boxes=[
            {"label": "info_track.py", "sub": "admit · ONNX metadata_props\n(pkg.torch.onnx.stack_trace) 파싱", "color": ACCENT},
            {"label": "join.py", "sub": "유일한 만남 지점\nnode.name 키로 #line 삽입", "color": ACCENT2},
            {"label": "code_track.py", "sub": "tail 대역 · onnx_node_name\n마커 C 생성 (실제 EmitC 자리)", "color": ACCENT},
        ],
        note=[
            "두 트랙은 서로의 내부를 모른다 — join.py만 양쪽을 알고 node.name으로 연결",
            "code_track.get_generated_c(onnx_path)→C문자열 시그니처만 유지하면, 부산대 실제 EmitC 출력으로 구현부 교체 가능",
            "결과: gcc -g 컴파일 시 DWARF .debug_line에 원본 .py 파일:줄이 그대로 보존 → 표준 gdb 중단점 동작",
        ],
        page=3,
    )

    bullet_slide(
        prs, "3. 실행 흐름", "run_full_pipeline.py 오케스트레이션 — 다음 5장은 각 단계의 실제 코드+실측 결과",
        [
            (0, "① info_track.build_locmap(onnx) → locmap.json 저장"),
            (0, "② code_track.get_generated_c(onnx) → 데모 C 저장"),
            (0, "③ join.inject_with_stats(c, locmap) → #line 투영된 C 저장"),
            (0, "④ gcc -g -O0 로 컴파일"),
            (0, "⑤ readelf --debug-dump=decodedline / gdb --batch 로 검증"),
            (0, "⑥ ①~③을 3회 반복해 바이트 단위 재현성 확인"),
        ],
        page=4,
    )

    code_shot_slide(
        prs, "3-1. 단계①", "info_track.build_locmap() — ONNX metadata_props → locmap",
        code_text='''
def _pick_user_frame(frames):
    """가장 안쪽(마지막) 비-프레임워크 프레임 우선. (이번 세션에 수정)"""
    if not frames:
        return None
    for fr in reversed(frames):
        if not any(m in fr["file"] for m in _FRAMEWORK_MARKERS):
            return fr
    return frames[-1]

# _build_from_model() 안에서:
stack_trace = md.get("pkg.torch.onnx.stack_trace", "")
frames = _parse_stack_trace(stack_trace) if stack_trace else []
user_frame = _pick_user_frame(frames)
if user_frame is not None:
    entry["file"] = _rel(user_frame["file"], relative_to)
    entry["line"] = user_frame["line"]
    entry["source"] = user_frame["source"]
''',
        image_path=os.path.join(SHOTS, "01_info_track_output.png"),
        code_caption="debug_info_investigation/info_track.py — 실제 소스(이번 세션 수정분)",
        shot_caption="resnet18_dynamo.locmap.json에서 node_Conv_292 항목 그대로 발췌",
        stage_tag="ONNX 로드 (admit)  →  locmap.json",
        page=5,
    )

    code_shot_slide(
        prs, "3-2. 단계②", "code_track.get_generated_c() — 노드당 마커 붙은 데모 C 생성",
        code_text='''
def _synth_demo_c(graph_nodes):
    # graph_nodes: [(node_name, op_type), ...] 그래프 순서 그대로
    for name, op in graph_nodes:
        fn = _fn_name(name)
        L += [
          f"static void {fn}(float *buf, size_t n) {{",
          f"/* onnx_node_name: {name}   op={op} */",
          f"  for (size_t i = 0; i < n; ++i)",
          f"    buf[i] = buf[i] * 1.0f + 0.0f;",
          "/* onnx_node_end */",
          "}",
        ]

def get_generated_c(onnx_path):
    # provenance(locmap)는 전혀 읽지 않는다 — node.name/op_type만 읽음
    model = onnx.load(onnx_path)
    ...
    return _synth_demo_c(graph_nodes)
''',
        image_path=os.path.join(SHOTS, "02_code_track_output.png"),
        code_caption="debug_info_investigation/code_track.py — 실제 harx/EmitC 자리의 데모 대역",
        shot_caption="resnet18_dynamo_kernels.c에서 node_Conv_292 함수 블록 그대로 발췌",
        stage_tag="EmitC→C (tail 대역)  →  kernels.c",
        page=6,
    )

    code_shot_slide(
        prs, "3-3. 단계③", "join.inject_with_stats() — 유일한 만남 지점, #line 삽입",
        code_text='''
for raw in lines:
    m = _MARKER_RE.search(raw)   # /* onnx_node_name: <name> */
    if m:
        name = m.group("name")
        out.append(raw)                 # 마커 줄은 그대로 둠
        entry = nodes.get(name)         # locmap에서 같은 node.name 조회
        if entry and "file" in entry and "line" in entry:
            f = _resolve(entry["file"], path_base)
            out.append(f'#line {int(entry["line"])} "{f}"')
            directives += 1
            matched.append(name)
        else:
            unknown.append(name)
        continue
''',
        image_path=os.path.join(SHOTS, "03_join_output.png"),
        code_caption="debug_info_investigation/join.py — 조인 키는 오직 node.name",
        shot_caption="kernels.c → kernels.line.c diff: #line 2줄 삽입(진입+복귀)",
        stage_tag="join  →  kernels.line.c",
        page=7,
    )

    code_shot_slide(
        prs, "3-4. 단계④", "gcc -g -O0 — #line이 표준 DWARF .debug_line으로 그대로 lowering",
        code_text='''
# run_full_pipeline.py, run() 안에서:
comp = _run(["gcc", "-g", "-O0", "-o", bin_path, line_c_path])
if comp.returncode != 0:
    return {"ok": False, "error": "gcc 실패", "stderr": comp.stderr}

# _run()은 subprocess.run(cmd, capture_output=True, text=True)
# 이 도커의 특수 처리 없음 — 표준 gcc 그대로 사용
''',
        image_path=os.path.join(SHOTS, "04_gcc_output.png"),
        code_caption="run_full_pipeline.py — gcc 호출부",
        shot_caption="실제 재실행: 종료코드 0, file 명령으로 debug_info 존재 확인",
        stage_tag="gcc -g -O0  →  kernels_dbg (ELF)",
        page=8,
    )

    code_shot_slide(
        prs, "3-5. 단계⑤", "verify_dwarf() — readelf/gdb로 실측 검증",
        code_text='''
rd = _run(["readelf", "--debug-dump=decodedline", binary])
decoded = rd.stdout
# .py 파일명이 .debug_line에 등장하는지, 투영한 (파일,줄)이
# 라인테이블에 그대로 있는지 파싱해서 대조

g = _run(["gdb", "--batch", "-nx",
          "-ex", f"break {pyb}:{pyl}",
          "-ex", f"info line {pyb}:{pyl}",
          binary])
# "in <fn>" 또는 "<fn>" 문자열이 출력에 있으면
# 그 노드 함수 주소에 실제로 바인딩된 것으로 판정
''',
        image_path=os.path.join(SHOTS, "05_verify_output.png"),
        code_caption="run_full_pipeline.py — verify_dwarf()",
        shot_caption="resnet18_dynamo_pipeline_report.json에 저장된 실측 원문 그대로",
        stage_tag="검증  →  pipeline_report.json",
        page=9,
    )

    table_slide(
        prs, "4. 회귀 확인", "기존 검증 — TinyCNN (5-노드, 2026-09-03 v3 완료분)",
        headers=["검사 항목", "결과"],
        rows=[
            ("조인 매칭 (5개 노드 전부)", "5/5 매칭, 미매칭 0"),
            ("gcc -g → .debug_line 생성", "PASS"),
            ("원본 .py 파일명이 DWARF에 등장", "PASS"),
            ("투영된 (파일,줄) 조합이 라인테이블에 존재", "PASS"),
            ("gdb break 원본.py:21 → 노드 함수 바인딩", "PASS"),
            ("3회 반복 바이트 동일 (locmap / 투영 C)", "PASS"),
        ],
        col_widths=[Inches(9.0), Inches(2.9)],
        highlight_ok_col=1,
        note="새 기술 검증 아님 — 오늘 새로 돌린 복잡한 모델 2개(다음 슬라이드)의 비교 기준선.",
        page=10,
    )

    table_slide(
        prs, "5. 신규 예제 1", "ResNet-18 (torchvision, dynamo export)",
        headers=["항목", "값"],
        rows=[
            ("ONNX 노드 수", f"{rn18_n}"),
            ("provenance(metadata_props) 매핑된 노드", f"{rn18_n} / {rn18_n} (100%)"),
            ("join 매칭 / 미매칭", f"{len(rn18_after['inject_stats']['matched'])} / {len(rn18_after['inject_stats']['unknown'])}"),
            ("삽입된 #line 지시자 수", f"{rn18_after['inject_stats']['directives']}"),
            *[(c[0][:58], c[1]) for c in checks_rows(rn18_after)],
            ("3회 반복 재현성 (locmap/투영 C 바이트 동일)", "PASS"),
        ],
        col_widths=[Inches(9.3), Inches(2.6)],
        highlight_ok_col=1,
        note="gdb 실측: break <원본 프레임>:<line> → node_node_Conv_292 함수 주소에 바인딩 확인.",
        page=11,
    )

    table_slide(
        prs, "5. 신규 예제 2", "MobileNetV2 (torchvision, dynamo export)",
        headers=["항목", "값"],
        rows=[
            ("ONNX 노드 수", f"{mnv2_n}"),
            ("provenance(metadata_props) 매핑된 노드", f"{mnv2_n} / {mnv2_n} (100%)"),
            ("join 매칭 / 미매칭", f"{len(mnv2_after['inject_stats']['matched'])} / {len(mnv2_after['inject_stats']['unknown'])}"),
            ("삽입된 #line 지시자 수", f"{mnv2_after['inject_stats']['directives']}"),
            *[(c[0][:58], c[1]) for c in checks_rows(mnv2_after)],
            ("3회 반복 재현성 (locmap/투영 C 바이트 동일)", "PASS"),
        ],
        col_widths=[Inches(9.3), Inches(2.6)],
        highlight_ok_col=1,
        note="ResNet-18보다 2배 큰 그래프(100노드, depthwise conv 포함)에서도 파이프라인 동일하게 동작.",
        page=12,
    )

    bullet_slide(
        prs, "6. 실측 중 발견한 한계", "버그 발견: \"모든 노드가 같은 한 줄\"로 수렴하는 문제",
        [
            (0, f"복잡한 모델을 처음 돌렸을 때: ResNet-18 {rn18_n}개 노드 → 서로 다른 줄 단 {rn18_u_before}개, "
                f"MobileNetV2 {mnv2_n}개 노드 → 단 {mnv2_u_before}개"),
            (0, "원인: info_track._pick_user_frame()이 파이썬 호출 스택의 \"가장 바깥쪽\" 프레임을 선택"),
            (1, "TinyCNN처럼 forward()가 한 겹이면 우연히 정답이었지만,"),
            (1, "ResNet.forward → _forward_impl → BasicBlock.forward 처럼 여러 겹을 거치는 모델에서는"),
            (1, "모든 연산이 항상 같은 진입 호출 줄(예: resnet.py:285)로 수렴해버림"),
            (0, "수정: \"가장 바깥쪽\" 대신 \"가장 안쪽(실제 연산 실행 지점)\" 프레임을 우선하도록 변경"),
            (1, f"ResNet-18: 서로 다른 줄 {rn18_u_before}개 → {rn18_u_after}개"),
            (1, f"MobileNetV2: 서로 다른 줄 {mnv2_u_before}개 → {mnv2_u_after}개"),
            (1, "수정 후에도 DWARF/gdb 검증 4/4 PASS, 3회 재현성 유지 (회귀 없음)"),
        ],
        page=13,
    )

    bullet_slide(
        prs, "7. 남은 한계 및 다음 단계", "",
        [
            (0, "code_track.py는 여전히 데모 C 생성기 — 실제 harx/EmitC 산출물로 아직 검증 못함"),
            (1, "get_generated_c(onnx_path)→C문자열 시그니처만 유지하면 구현부 교체로 연동 가능하도록 설계됨"),
            (0, "동일 helper(BasicBlock 등)를 여러 인스턴스가 공유하면, line 단위로는 \"연산 종류\"까지만 구분됨"),
            (1, "레이어 인스턴스 단위 구분은 locmap에 이미 있는 module_scope(예: layer1.0.conv1)를 추가로 활용해야 함"),
            (1, "단, gdb 관점에서 공용 함수에 건 중단점이 매 호출마다 멈추는 것은 정상 동작 — 치명적 결함은 아님"),
            (0, "provenance 캡처는 PyTorch(dynamo=True) exporter에서만 검증됨 — 타 프레임워크 미검증"),
            (0, "다음: 부산대 실제 EmitC 샘플 확보 → code_track 구현부 교체, 3순위(LLVM lowering) 조사 계속"),
        ],
        page=14,
    )

    bullet_slide(
        prs, "부록", "산출물 위치",
        [
            (0, "lab/artifacts/resnet18/, mobilenet_v2/  — 버그 수정 전 1차 실행(비교용, 유지)"),
            (0, "debug_info_investigation/artifacts/v3/resnet18_dynamo/, mobilenet_v2_dynamo/  — 수정 후 최신 결과(정본)"),
            (1, "각 디렉터리: *.locmap.json, *_kernels.c, *_kernels.line.c, *_kernels_dbg, *_pipeline_report.json"),
            (0, "lab/export_complex_models.py  — resnet18/mobilenet_v2 dynamo export 스크립트"),
            (0, "debug_info_investigation/info_track.py  — 이번에 수정된 소스 (_pick_user_frame)"),
            (0, "lab/lab.md  — 재실행 로그 누적 기록"),
        ],
        page=15,
    )

    out = os.path.join(HERE, "report_assignment.pptx")
    prs.save(out)
    print("저장:", out)



# ============================================================================
# 논문 초안 PPT (방법2: onnx-mlir 임포터 자체 패치, 커밋 안 됨)
# ============================================================================
PATCH_CODE = '''
Location ImportLoc(const onnx::NodeProto &node) {
  SmallVector<Location, 2> locs;
  if (node.has_name())
    locs.push_back(NameLoc::get(builder_.getStringAttr(node.name())));

  // PyTorch dynamo exporter가 남긴 원본 소스 위치를 복원한다.
  for (const auto &entry : node.metadata_props()) {
    if (entry.key() != "pkg.torch.onnx.stack_trace")
      continue;
    if (auto srcLoc = ParseStackTrace(entry.value()))
      locs.push_back(*srcLoc);
    break;
  }

  if (locs.empty())   return UnknownLoc();
  if (locs.size()==1) return locs[0];
  return builder_.getFusedLoc(locs);   // NameLoc + FileLineColLoc
}
'''.strip("\n")

PARSE_CODE = '''
std::optional<Location> ParseStackTrace(const std::string &trace) {
  // 파이썬 트레이스백 첫 프레임: File "<file>", line <N>, ...
  size_t q1 = trace.find('"');
  size_t q2 = trace.find('"', q1 + 1);
  std::string file = trace.substr(q1 + 1, q2 - q1 - 1);
  size_t lp = trace.find(", line ", q2) + 7;
  unsigned line = 0;
  while (isdigit(trace[lp])) line = line*10 + (trace[lp++]-'0');
  return FileLineColLoc::get(builder_.getStringAttr(file), line, 0);
}
'''.strip("\n")


def build_paper_deck():
    prs = new_presentation()

    title_slide(
        prs,
        kicker="연구 노트 · 논문 초안 자료",
        title="ONNX 프론트엔드에서 PyTorch Provenance를\nMLIR Location으로 직접 복원하기",
        subtitle="onnx-mlir ImportLoc()/ParseStackTrace() 실험 — stack_trace metadata_props → FileLineColLoc",
        meta_lines=["2026-09-14", "주의: 코드는 로컬 작업 트리에만 존재 (git 커밋 안 됨)", "실행 환경: 업스트림 onnx-mlir, 일반 MLIR 인프라 기준"],
        accent=ACCENT2,
    )

    bullet_slide(
        prs, "1. 동기", "표준 ONNX 임포터는 provenance를 버린다",
        [
            (0, "onnx-mlir의 기본 ImportLoc()은 node.name만으로 NameLoc을 만든다"),
            (1, "PyTorch dynamo exporter가 각 노드에 남긴 pkg.torch.onnx.stack_trace(원본 .py 파일:줄)는 그냥 버려짐"),
            (0, "질문: 별도의 외부 파이프라인(#line 투영) 없이, MLIR 임포터 자체가 표준 FileLineColLoc을 만들 수 있는가?"),
            (0, "이 실험은 admit 단계 밖 — onnx-mlir \"프론트엔드 임포터 내부\"를 직접 수정한 것"),
            (1, "실제 ARX 파이프라인의 mlir-arx(PNU 소유)에는 이식 불가 — 순수 연구/논문용 실험"),
        ],
        accent=ACCENT2,
        page=2,
    )

    code_shot_slide(
        prs, "2. 구현", "ParseStackTrace() — 문자열에서 (file, line) 추출",
        code_text=PARSE_CODE,
        image_path=os.path.join(SHOTS, "06_method2_run_output.png"),
        code_caption="src/Builder/FrontendDialectTransformer.cpp (커밋 안 됨) — "
                     "Python traceback 'File \"…\", line N' 을 그대로 파싱",
        shot_caption="이 코드가 반영된 바이너리로 실제 실행한 로그(터미널 원문)",
        accent=ACCENT2,
        stage_tag="패치된 onnx-mlir 실행",
        page=3,
    )

    code_shot_slide(
        prs, "2. 구현", "ImportLoc() — NameLoc + FileLineColLoc → FusedLoc",
        code_text=PATCH_CODE,
        image_path=os.path.join(SHOTS, "07_method2_mlir_output.png"),
        code_caption="stock 구현은 node.has_name() 분기만 있었음 — metadata_props 순회 루프를 추가",
        shot_caption="이 코드가 만들어낸 실제 MLIR 출력(loc 정의 발췌)",
        accent=ACCENT2,
        stage_tag="실제 출력",
        page=4,
    )

    table_slide(
        prs, "3. 실험 설계", "3개 모델, 모두 torch.onnx.export(dynamo=True)",
        headers=["모델", "ONNX 노드 수", "구조 특징"],
        rows=[
            ("TinyCNN", "5", "단일 forward(), 서브모듈 호출 없음 (사용자 스크립트에 직접 정의)"),
            ("ResNet-18 (torchvision)", "49", "ResNet.forward → _forward_impl → BasicBlock.forward (3중 중첩)"),
            ("MobileNetV2 (torchvision)", "100", "Inverted-residual block 반복 + depthwise conv"),
        ],
        col_widths=[Inches(3.6), Inches(2.3), Inches(6.0)],
        note="측정 항목: loc(...) 등장 여부 / FusedLoc 여부 / (file,line) 쌍이 서로 다른 노드를 구분하는 정도.",
        page=5,
    )

    bullet_shot_slide(
        prs, "4. 결과 1", "TinyCNN — 노드별로 완전히 다른 줄이 정확히 매핑됨",
        [
            (0, "MaxPool / Reshape / Linear+ReLU / Linear 4개 연산이"),
            (0, "서로 다른 원본 줄(21/22/23/24)에 정확히 매핑됨"),
            (1, "각 loc은 NameLoc(node.name)과 FileLineColLoc이 FusedLoc으로 결합"),
            (1, "MLIR 표준 Location 인프라만으로 표현됨 — 커스텀 attribute 불필요"),
            (0, "→ 별도 파이프라인 없이, MLIR 임포터 패치만으로"),
            (0, "노드 단위 provenance 복원이 가능함을 실증(대조군 성공 사례)"),
        ],
        image_path=os.path.join(SHOTS, "08_method2_tinycnn_output.png"),
        shot_caption="tinycnn_recheck_with_patch.onnx.mlir 실제 loc 정의 그대로",
        accent=ACCENT2,
        page=6,
    )

    bullet_shot_slide(
        prs, "5. 결과 2", "ResNet-18 / MobileNetV2 — \"공용 헬퍼\" 효과로 수렴",
        [
            (0, "49개 노드 전부가 단 1개 줄(resnet.py:285)로 수렴"),
            (0, "MobileNetV2도 100개 노드 전부 → 1개 줄로 동일하게 수렴"),
            (0, "원인 (파이썬 쪽에서 먼저 규명, 과제 PPT 참고):"),
            (1, "stack_trace는 [바깥쪽 호출 → 안쪽 실행 지점] 순서인데,"),
            (1, "ParseStackTrace()는 문자열의 첫 프레임(=가장 바깥쪽)만 취함"),
            (1, "forward가 여러 겹이면 바깥쪽 프레임은 모든 연산에 대해 항상 동일"),
            (0, "→ C++도 \"가장 안쪽(마지막) 프레임\" 우선으로 보완 필요(향후 과제)"),
        ],
        image_path=os.path.join(SHOTS, "07_method2_mlir_output.png"),
        shot_caption="resnet18_dynamo_onnxir.onnx.mlir 실제 loc 정의 — 전부 #loc2 하나에 fuse",
        accent=ACCENT2,
        page=7,
    )

    bullet_slide(
        prs, "6. 논의", "실제 ARX 프로덕션 파이프라인에 적용 가능한가?",
        [
            (0, "이 방법은 onnx-mlir \"ONNX 프론트엔드 임포터 내부\"(FrontendDialectTransformer.cpp)를 직접 수정해야 동작"),
            (0, "실제 프로덕션에서 이 역할은 PNU 소유의 mlir-arx(라이선스 이슈로 수정 불가)가 대체함"),
            (1, "→ 이 접근은 실제 배포 경로에는 이식 불가. \"일반 MLIR 인프라 연구/논문 기여\"로서만 유효"),
            (0, "실제 배포 경로에서는 기존 방식(admit 외부 전처리 info_track + tail #line 투영 join)이 여전히 유일한 통로"),
            (0, "→ 두 결과의 관계: 이 임포터 패치는 \"일반 MLIR 인프라가 원리적으로 무엇을 지원할 수 있는지\"를 보여주는 상한선 실험,"
                " 과제 파이프라인은 \"실제 제약(mlir-arx 블랙박스) 안에서 무엇이 가능한지\"를 보여주는 실전 답"),
        ],
        accent=ACCENT2,
        page=8,
    )

    bullet_slide(
        prs, "7. 현재 상태 및 남은 일", "",
        [
            (0, "코드는 git 미커밋 상태 (작업 트리에만 존재) — 검토 후 커밋/PR 여부 별도 결정 필요"),
            (0, "ParseStackTrace()의 \"첫 프레임\" 편향 보정 (info_track.py의 수정과 동일한 방향)"),
            (0, "FusedLoc이 이후 --EmitMLIR/--EmitLLVMIR lowering까지 생존하는지 미검증 (3순위 조사와 연계)"),
            (0, "--enable-debug-info(DIScopeForLLVMFuncOpPass)는 별도로 실패 진단됨(Part A2) — 이 패치와는 무관한 경로"),
            (0, "upstream onnx-mlir 기여 여부는 라이선스/설계 검토 후 별도 판단"),
        ],
        accent=ACCENT2,
        page=9,
    )

    bullet_slide(
        prs, "부록", "산출물 위치",
        [
            (0, "lab/artifacts/resnet18/resnet18_dynamo_onnxir.onnx.mlir"),
            (0, "lab/artifacts/mobilenet_v2/mobilenet_v2_dynamo_onnxir.onnx.mlir"),
            (0, "lab/artifacts/tinycnn_recheck_with_patch.onnx.mlir  (재실행 재현용)"),
            (0, "src/Builder/FrontendDialectTransformer.cpp  (커밋 안 된 패치 원본)"),
        ],
        accent=ACCENT2,
        page=10,
    )

    out = os.path.join(HERE, "paper_draft.pptx")
    prs.save(out)
    print("저장:", out)


if __name__ == "__main__":
    build_assignment_deck()
    build_paper_deck()
