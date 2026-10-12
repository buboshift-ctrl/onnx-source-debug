# 작업 지시: 검증된 두 트랙 구조를 실제 코드 구조로 정리 (v3)

- 전제 문서(반드시 먼저 참고): `CLAUDE.md`, `debug_info_investigation.md`, `TASK_implement_dwarf_pipeline_v2.md`(완료됨, 삭제하지 말 것)
- **이 문서는 새로운 기술 검증이 아니다.** v2에서 이미 검증된 결론(Part A/B 성공, A2 실패)을 재논쟁하지 않는다. 목적은 "동작하는 데모 스크립트"를
  "구조가 명확해서 나중에 실제 harx/EmitC가 연결될 때 갈아끼우기만 하면 되는 코드"로 **리팩터링**하는 것.

## 확정된 아키텍처 (이번에 정리된 것, 그대로 코드에 반영할 것)

```
                 [ONNX 파일 — metadata_props 포함, 어느 쪽도 안 건드림]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      정보 트랙 (우리 코드)          코드 트랙 (외부, 건드리지 않음)
   locmap_capture.py                onnx-mlir → harx → EmitC
   metadata_props 읽음                node.name만 유지한 채 통과
              │                           │
              ▼                           ▼
        locmap.json                  생성된 C 코드
   node.name → 원본 .py 파일:줄        node.name 흔적만 남음
              │                           │
              └─────────────┬─────────────┘
                            ▼
                inject_line_directives.py
              (join key = node.name, #line 삽입)
                            │
                            ▼
                   gcc -g → 실행파일
              (진짜 DWARF, gdb로 원본 .py 줄 중단점)
```

**핵심 설계 원칙 (코드에도 이대로 드러나야 함)**:
- 정보 트랙과 코드 트랙은 서로의 존재를 몰라도 되는 완전히 독립된 모듈이어야 한다.
- 두 트랙이 만나는 지점은 딱 하나, `join` 단계뿐이고, 조인 키는 오직 `node.name` 하나다.
- 코드 트랙은 지금 이 컨테이너에 실제 harx/EmitC가 없어서 **데모 생성기로 대체돼 있다**. 이 대체 가능성이 코드 구조에서
  명확히 드러나야 한다 — 나중에 실제 harx/EmitC 출력을 넣을 때 이 부분 함수 하나만 바꾸면 되도록.

## 구현 항목

### 1. 코드 트랙의 "인터페이스"를 명시적으로 분리

현재 `inject_line_directives.py --demo`가 데모 C 생성과 `#line` 삽입 로직을 한 파일에 같이 갖고 있을 텐데, 이걸 나눠라:

- `debug_info_investigation/code_track.py`
  - `get_generated_c(onnx_path: str) -> str` 함수 하나만 노출.
  - 지금은 내부에서 데모 C를 생성해서 리턴 (기존 `--demo` 로직 그대로 옮기기).
  - 함수 docstring에 **"실제 harx/EmitC 연동 시 이 함수의 구현부만 교체하면 됨. 함수 시그니처(입력: onnx 경로, 출력: C 소스 문자열)는
    유지할 것"**이라고 명시.

- `debug_info_investigation/info_track.py` (기존 `locmap_capture.py`를 이 이름으로 정리하거나, 얇은 wrapper로 감싸기)
  - `build_locmap(onnx_path: str) -> dict` 함수로 정리.

- `debug_info_investigation/join.py` (기존 `inject_line_directives.py`의 조인 로직만 남기기)
  - `inject(c_source: str, locmap: dict) -> str` 함수로 정리. `node.name` 조인, `#line` 삽입 로직.

### 2. 오케스트레이터 스크립트

`debug_info_investigation/run_full_pipeline.py`:
```
locmap = info_track.build_locmap(onnx_path)
c_src = code_track.get_generated_c(onnx_path)
annotated = join.inject(c_src, locmap)
# 컴파일 + readelf/gdb 검증까지 기존 run_pipeline.sh 로직 재사용
```
기존 `run_pipeline.sh`가 이미 하던 일(컴파일, readelf, gdb 검증, 3회 반복 재현성 체크)을 이 오케스트레이터가 호출하도록 정리.
`run_pipeline.sh`는 지우지 말고, 이 파이썬 오케스트레이터를 부르는 얇은 셸로 남겨도 됨.

### 3. 리팩터링 후 회귀 확인 (필수)

구조만 바꾸는 거라 **동작이 달라지면 안 된다.** 리팩터링 후 반드시:
- `readelf --debug-dump=decodedline`에서 v2 때와 동일하게 원본 `.py:21..24`가 나오는지 재확인
- `gdb break inspect_export_metadata.py:21`이 동일하게 바인딩되는지 재확인
- 3회 반복 바이트 동일성 재확인
- 이 중 하나라도 v2 때와 달라지면 리팩터링이 로직을 건드린 것이니 원인을 찾아 고칠 것 (새 버그를 "구조 변경 때문"이라고 넘기지 말 것)

### 4. CLAUDE.md에 알려진 제약사항 추가

다음 섹션이 없으면 추가할 것:
```markdown
## 알려진 범위 제한
locmap 캡처(stack_trace 기반)는 PyTorch(torch.onnx.export dynamo=True)에서만 검증됨.
TensorFlow/scikit-learn 등 다른 프레임워크의 ONNX 변환기가 동등한 metadata_props를 채우는지는 미확인.
2026년 검증 벤치마크(ResNet-18, MobileNetV2)가 전부 TorchVision 기반이라 당장은 문제 없지만,
다른 프레임워크 모델이 들어오면 이 가정을 재검증해야 함.

코드 트랙(code_track.py)은 현재 데모 C 생성기로 대체돼 있음 — 실제 harx/EmitC 출력이 아님.
부산대로부터 실제 EmitC 샘플을 받으면 code_track.get_generated_c()의 구현부만 교체하면 되도록 설계함.
```

## Definition of Done

- [ ] `info_track.py` / `code_track.py` / `join.py` / `run_full_pipeline.py`로 구조가 정리됨
- [ ] `code_track.get_generated_c()`가 향후 실제 harx/EmitC 연동 지점이라는 게 docstring에 명시됨
- [ ] 리팩터링 후에도 readelf/gdb 검증 결과가 v2와 동일함 (회귀 없음)
- [ ] CLAUDE.md에 "알려진 범위 제한" 섹션 추가됨
- [ ] `debug_info_investigation.md`에 "구조 정리(v3)" 섹션 추가, 기존 내용 삭제 금지

## 하지 말 것

- 이번엔 새로운 기술적 결론을 만들지 않는다 — 순수 리팩터링. Part A2(진짜 DWARF)를 다시 시도하지 말 것.
- `code_track.py`의 데모 생성기 로직 자체를 "개선"하려 하지 말 것 — 구조만 옮기고 동작은 그대로 유지.
- onnx-mlir/`third_party` 수정 금지 (v2와 동일)
