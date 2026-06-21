# Synesthesia — 프로젝트 정리 문서

> 이미지 한 장을 넣으면, 그 이미지에서 영감을 받은 **노래(보컬+반주)** 와 **앨범 커버**를
> 만들어내는 라이브 데모. 세 개의 HuggingFace foundation model이 하나의 task에 협력한다.

---

## 1. 과제 요건과 충족 방법

| 요건 | 충족 |
|---|---|
| foundation model 3개 이상 | Qwen2.5-VL + ACE-Step + Stable Diffusion (서로 다른 패밀리·모달리티) |
| 하나의 task | 이미지 → 노래 + 앨범 커버 |
| 라이브 데모 | 브라우저에서 이미지 업로드 → 모델별 결과가 실시간 스트리밍 |
| 발표 당일 촬영 이미지 불필요 | 어떤 이미지든 즉석 업로드 가능 |
| 재현 가능한 repo + README | 2개 conda 환경 셋업·실행법 명시 (README.md) |
| 전부 로컬 (API 미사용) | 모든 가중치를 HuggingFace Hub에서 받아 로컬 GPU로 추론 |

---

## 2. 무엇을 하는가 (Task)

입력: **이미지 1장**
출력:
- **노래** `song.wav` — 가사가 실제로 **불러지는** 곡 (보컬 + 반주, 25초, 48kHz 스테레오)
- **앨범 커버** `cover.png` — 곡 분위기를 담은 512×512 이미지

---

## 3. 아키텍처

```
                    image
                      │
            ┌─────────▼──────────┐
   ①        │   Qwen2.5-VL-3B    │  본다 + 쓴다
            │ (vision+language)  │
            └─────────┬──────────┘
        { title, mood, genre, tempo,
          lyrics([verse]/[chorus]),
          tags, cover_prompt }
              │                  │
       lyrics+tags          cover_prompt
              │                  │
     ┌────────▼───────┐  ┌───────▼─────────┐
  ②  │  ACE-Step-3.5B │  │  SDXL-Turbo     │  ③
     │ (music+voice)  │  │ (image gen)     │
     └────────┬───────┘  └───────┬─────────┘
          song.wav            cover.png
```

| # | 모델 (HF) | 모달리티 | 역할 |
|---|---|---|---|
| ① | `Qwen/Qwen2.5-VL-3B-Instruct` | 비전 + 언어 | 이미지의 무드를 읽고 가사 + 스타일 태그 + 커버 프롬프트 작성 |
| ② | `ACE-Step/ACE-Step-v1-3.5B` | 음악 + 가창 | 가사 + 태그 → 보컬이 들어간 곡 전체 생성 |
| ③ | `stabilityai/sdxl-turbo` | 이미지 생성 | 커버 프롬프트 → 앨범 커버 |

---

## 4. 설계 결정 (왜 이렇게)

- **Qwen2.5-VL이 보기+쓰기를 모두 담당**: 비전 모델을 따로(BLIP-2) 두고 텍스트 LLM을
  또 두면 "VL이 있는데 왜 안 썼냐"는 비판을 받는다. VL 하나가 보기+작사를 다 하니
  그 비판이 성립하지 않고, 세 모델의 역할이 깔끔히 분리된다.
- **ACE-Step으로 진짜 "노래"**: 초기엔 MusicGen(반주) + Bark(보컬)를 썼으나 Bark는
  본질적으로 TTS라 노래처럼 들리지 않았다. ACE-Step은 가사+스타일을 받아 *보컬이 실제로
  불러지는* 곡을 만든다.
- **Stable Diffusion으로 앨범 커버**: 세 번째 모델이 음악과 겹치지 않는 *다른 모달리티*
  (이미지 생성)를 맡아 역할 중복이 없고, 데모의 시각적 임팩트가 커진다.

---

## 5. 기술 구현

- **스트리밍 UI**: 서버가 파이프라인을 단계별로 실행하며 NDJSON으로 흘려보내, 모델이
  하나 끝날 때마다 브라우저 카드(①②③)가 즉시 채워진다. "세 모델이 순서대로 일하는" 모습이
  눈에 보인다. (`server.py` + `web/index.html`)
- **VRAM 관리 (12GB)**: 각 모델은 자기 단계가 끝나면 GPU에서 해제된다
  (`config.FREE_AFTER_STAGE`). 덕분에 12GB 노트북 GPU 한 장에 다 들어간다.
- **2개 환경 격리 + 서브프로세스**: ACE-Step이 `transformers==4.50`을 핀하는데 이는
  Qwen2.5-VL이 쓰는 transformers 5.x와 충돌한다. 그래서 ACE-Step은 별도 `ace` 환경에
  설치하고 `ace_singer.py`를 **서브프로세스**로 호출한다. 메인 `synesthesia` 환경은
  Qwen2.5-VL과 Stable Diffusion을 담당한다.

---

## 6. 해결한 기술 난관

| 문제 | 해결 |
|---|---|
| ACE-Step `transformers==4.50` 핀이 Qwen2.5-VL(5.x)과 충돌 | 격리 `ace` 환경 + 서브프로세스 호출 |
| torchaudio 2.10이 저장을 TorchCodec으로 강제 (Windows 휠 없음) | `torchaudio.save`를 soundfile로 몽키패치 (`ace_singer.py`) |
| PyPI `ace-step` 0.1.0 패키지 깨짐 (sdist에 requirements.txt 누락, 한글 로케일 cp949 디코드 오류) | GitHub에서 `PYTHONUTF8=1`로 설치 |
| Qwen2.5-VL의 JSON 출력이 잘리거나 가사의 `[verse]` 대괄호가 JSON 파싱을 깨뜨림 | 토큰 한도 상향(1024) + 정규식 salvage 파서 폴백 |
| 노트북 GPU가 유휴 시 180MHz로 표시 | 부하 시 2812MHz로 정상 부스트 (유휴 클럭일 뿐, 잠김 아님) |

---

## 7. 실행 방법 (요약)

상세 셋업은 `README.md` 참조. 핵심만:

```bash
# 메인 환경 (Qwen2.5-VL + Stable Diffusion)
conda activate synesthesia
python server.py            # http://localhost:8000 접속 → 이미지 업로드

# ACE-Step은 별도 ace 환경에 설치되어 서브프로세스로 자동 호출됨
```

CLI: `python demo.py examples/sunset.jpg --out outputs/cli`

가중치는 첫 실행 시 HuggingFace Hub에서 자동 다운로드 후 캐시된다.

---

## 8. 파일 구조

```
synesthesia/
  server.py          # FastAPI 웹앱 (단계별 스트리밍)
  web/index.html     # 브라우저 UI: 업로드 + 모델별 카드 + 오디오/커버
  demo.py            # CLI 진입점
  ace_singer.py      # ACE-Step 실행 스크립트 (ace 환경에서 동작)
  config.py          # 모델 ID + 하이퍼파라미터
  pipeline/
    lyricist.py      # ① Qwen2.5-VL
    singer.py        # ② ACE-Step (서브프로세스)
    cover.py         # ③ Stable Diffusion
    manager.py       # VRAM 헬퍼
    __init__.py      # 오케스트레이터 (stream_song / image_to_song)
  requirements.txt
  README.md          # 셋업/실행 가이드
  REPORT.md          # 본 문서
  examples/sunset.jpg
```

---

## 9. 한계 및 향후

- 가창 품질은 ACE-Step의 스타일 태그와 디퓨전 스텝(`ACE_INFER_STEP`)에 좌우된다.
  스텝을 올리면 품질↑·속도↓.
- 곡 길이는 `ACE_DURATION_S`(기본 25초). 길이를 늘리면 생성 시간이 늘어난다.
- 한국어 가사도 ACE-Step이 지원하므로, 프롬프트를 바꾸면 한국어 곡으로 확장 가능.

---

## 10. 발표 시나리오 (3분 예시)

1. (0:00) "이미지 한 장으로 노래와 앨범 커버를 만드는 Synesthesia입니다." — 컨셉 한 줄.
2. (0:20) 이미지 업로드 → ① Qwen2.5-VL이 가사/스타일을 쓰는 카드가 채워짐.
3. (0:50) ② ACE-Step이 부른 곡을 **재생** — 핵심 임팩트.
4. (1:40) ③ Stable Diffusion 앨범 커버 공개.
5. (2:10) "비전언어·가창·이미지생성 세 foundation model이 하나의 곡을 함께 만들었습니다."
   — 요건 매핑으로 마무리.
