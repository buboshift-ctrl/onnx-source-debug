# 환경 스냅샷 — 이 세션이 실행된 컨테이너 상태

**중요**: 이 도커 환경(패키지 설치 상태, 사용자 계정, venv 유무 등)은 **세션마다 달라질 수 있다.**
`lab.md` 1절(2026-09-11 작성)은 그 세션의 컨테이너를 기준으로 "`/home/ljy/venv`를 반드시 써야
한다"고 적었지만, 이 문서를 쓴 세션(아래 타임스탬프)의 컨테이너에는 그 사용자/venv 자체가
존재하지 않았고, 시스템 `python3`에 필요한 패키지가 이미 전역 설치되어 있었다. **과거 기록의
"실행 방법"을 그대로 믿지 말고, 매 세션 시작 시 아래 확인 절차를 먼저 다시 실행할 것.**

## 확인 절차 (다음 세션에서 그대로 실행)

```bash
whoami; id
python3 --version
python3 -c "import torch; print(torch.__version__, torch.__file__)"
python3 -c "import torchvision; print(torchvision.__version__)"
python3 -c "import onnx; print(onnx.__version__)"
python3 -c "import pptx; print(pptx.__version__)"
ls /home   # venv가 다른 사용자 홈 아래 있는지 확인
```

## 이 세션의 스냅샷 (2026-09-14T07:14:42Z)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| 컨테이너 hostname | `3be41e623fc1` |
| 사용자 | `root` (uid=0) — 2026-09-11 세션의 `ljy`(비-root)와 다름 |
| `/home` 내용 | `ubuntu` 사용자만 존재, `ljy` 없음 → `/home/ljy/venv` 자체가 없음 |
| `python3` | `/usr/bin/python3`, Python 3.12.3 (시스템 python, venv 아님) |
| `torch` | 2.14.0+cpu, `/usr/local/lib/python3.12/dist-packages/torch` (전역 설치, 이 세션 시작 전부터 이미 있었음) |
| `onnx` | 1.21.0, 전역 설치 |
| `torchvision` | 0.29.0+cpu — **이 세션에서 새로 설치** (아래 "설치 이력" 참고) |
| `python-pptx` | 1.0.2 — **이 세션에서 새로 설치** |
| `build/Debug/bin/onnx-mlir` | 2026-09-10 02:32 빌드 (이 세션이 새로 빌드한 것 아님, 이미 있던 바이너리) |
| `src/Builder/FrontendDialectTransformer.cpp` | git 기준 `M`(수정, 커밋 안 됨) — 위 바이너리에 이미 반영된 상태로 확인됨 |

## 이 세션에서의 패키지 설치 이력 (사용자 승인 후 실행)

```bash
# 1차 시도: 실패 (torchvision이 CUDA용 torch에 링크되어 있어 +cpu 빌드와 ABI 불일치)
pip3 install --break-system-packages --quiet torchvision python-pptx
# → python-pptx는 성공, torchvision import 시 "operator torchvision::nms does not exist" 에러

# 해결: PyTorch 공식 CPU 휠 인덱스에서 강제 재설치
pip3 install --break-system-packages --quiet --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cpu torchvision
# → torchvision 0.29.0+cpu 로 정상 동작 확인
```

**교훈**: `torch`가 `+cpu` 빌드로 설치되어 있으면, `torchvision`도 반드시 같은
`download.pytorch.org/whl/cpu` 인덱스에서 설치해야 한다. 기본 PyPI에서 받으면 CUDA용 torch에
링크된 torchvision이 설치되어 import 시점에 ABI 불일치로 깨진다.

## 추가 발견: 같은 대화 도중에도 컨테이너가 바뀔 수 있다 (2026-09-14, 같은 세션 안에서)

위 스냅샷을 적고 몇 시간 뒤(같은 대화, 세션 재시작 아님), `torch`/`torchvision`/`python-pptx`가
갑자기 전부 `ModuleNotFoundError`가 났다. `/workdir` 밑 파일들은 전부 그대로 있었는데
(수십 개 산출물, 방금 만든 스크립트 등 하나도 안 없어짐), 딱 pip로 설치한 패키지만 사라짐.
`hostname`을 찍어보니 `3be41e623fc1` → `f0b44b348067`로 바뀌어 있었다(uptime은 3일째로 정상 —
컨테이너 재부팅이 아니라 **다른 컨테이너로 라우팅된 것**). `onnx`는 살아남았는데(아마 이미지에
기본 포함) `torch`/`torchvision`/`python-pptx`는 이 세션에서 직접 pip install 한 것들이라
새 컨테이너엔 없었다.

**결론**: `/workdir`(git 리포)는 여러 컨테이너 인스턴스가 공유하는 볼륨이라 파일은 안전하지만,
`pip install`로 깐 패키지는 컨테이너 자신의 파일시스템에만 있어서 **대화 중간에도 사라질 수
있다.** 오래 걸리는 작업(특히 pip 설치가 필요한 스크립트를 여러 번 나눠 실행할 때)은:
- 패키지를 쓰는 명령 직전에 `python3 -c "import X"`로 매번 다시 확인할 것 — 조금 전에 됐다고
  이번에도 된다고 가정하지 말 것.
- 없어졌으면 이미 사용자 승인을 받았던 패키지라면 재확인 없이 그냥 다시 설치하고 계속 진행해도
  됨(새로운 설치 요청이 아니라 승인된 상태 복구이므로).

## 왜 이 문서를 만들었나

이전 세션(2026-09-11)이 `lab.md`에 남긴 "이 컨테이너엔 X가 없다/Y venv를 써야 한다"는 서술을,
이번 세션에서 그대로 믿고 찾다가 실제로는 다르다는 걸 뒤늦게 발견했다(사용자 질문으로 확인).
같은 혼란이 다음 세션에서도 반복되지 않도록, "코드/문서의 서술"이 아니라 "그 순간 직접 확인한
사실"만 스냅샷으로 남긴다. 다음 세션도 이 파일의 옛 스냅샷을 맹신하지 말고 위 확인 절차를 다시
실행해서 새 스냅샷을 이 파일에 추가(append)할 것.
