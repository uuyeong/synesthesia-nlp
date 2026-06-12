# 배포 가이드 — Hugging Face Spaces (24/7 공개 링크)

Synesthetic AI 데모를 누구나 링크만으로 24시간 접근할 수 있도록
Hugging Face Spaces(Gradio SDK)에 배포한다. 개인 PC가 꺼져 있어도 동작한다.

- **공개 링크:** https://huggingface.co/spaces/uuyeong/synesthetic-ai
- **사이트 이름:** Synesthetic AI (프로젝트 이름)
- **호스팅:** HF Spaces 무료 CPU(basic) 티어

---

## 한 번에 배포/갱신하기

```bash
hf auth login          # 최초 1회 — HF 토큰(write 권한) 입력
python scripts/deploy_hf.py
```

`scripts/deploy_hf.py` 가 하는 일:
1. 로그인 사용자명으로 Space `<user>/synesthetic-ai` 생성(있으면 재사용)
2. HF Space 카드용 `README.md`(YAML front matter)를 **동적 생성**해 업로드
   — 로컬/GitHub `README.md` 는 건드리지 않는다
3. 런타임에 필요한 파일만 선별 업로드(`ALLOW_PATTERNS`), 대용량 `.npy` 는 LFS 자동 처리

코드/데이터를 수정한 뒤 다시 `python scripts/deploy_hf.py` 만 실행하면 갱신된다.

---

## 업로드되는 파일 (런타임 필수)

| 경로 | git 추적 | 비고 |
|---|---|---|
| `src/*.py` | ✅ | 앱·파이프라인·시각화 |
| `requirements.txt` | ✅ | HF 가 빌드 시 설치 |
| `data/bert_mean_vec.npy`, `bert_anchor_R/G/B.npy` | ✅ | Mean Centering + 앵커 |
| `data/synesthesia_grapheme_mean_rgb.csv`, `nrc_word_rgb.csv` | ✅ | Eagleman / NRC |
| `data/poetry_candidate_words.txt` | ✅ | 역방향 후보 9,770개 |
| `data/candidate_vectors.npy` (~29MB) | ❌ gitignore | **역방향·코사인 보정·추천에 필수** — 없으면 9,770개 즉석 추론으로 사실상 타임아웃 |
| `data/mlp_weights.pt` (99KB) | ❌ gitignore | 없으면 rgb_syn→rgb_uni fallback |

> `data/candidate_vectors.npy` 와 `data/mlp_weights.pt` 는 `.gitignore` 대상이라
> GitHub 엔 없다. 이 스크립트는 **로컬 파일**을 직접 HF 로 올리므로, 배포 PC 에
> 두 파일이 실제로 존재해야 한다(없으면 스크립트가 사전 검사에서 중단).

---

## 동작 원리 메모

- HF 는 `app_file: src/app.py` 를 실행한다. `python src/app.py` 형태라 스크립트
  디렉터리(`src/`)가 `sys.path` 에 올라가 모듈 간 flat import 가 그대로 동작한다.
- `app.py` 는 `SPACE_ID` 환경변수로 HF 여부를 감지해, HF 에서는 `share=False`
  (플랫폼이 0.0.0.0:7860 자동 바인딩 + 영구 URL 제공), 로컬에서는 `share=True`.
- `sdk_version: 6.14.0` 으로 고정 — `demo.launch(theme=, css=)` 는 gradio 6.x 에서만
  유효하다(5.x 는 theme/css 인자 위치가 달라 실패할 수 있음).
- 첫 빌드는 `torch`/`transformers` 설치 + `bert-base-uncased` 다운로드로 수 분 소요.
  이후 기동은 빠르다.

---

## 상태 확인 / 문제 해결

```bash
# 빌드/실행 상태
python -c "from huggingface_hub import HfApi; print(HfApi().space_info('uuyeong/synesthetic-ai').runtime.stage)"
```

- `BUILDING` → 빌드 중, `RUNNING` → 접속 가능, `RUNTIME_ERROR`/`BUILD_ERROR` → Space
  웹페이지의 **Logs** 탭에서 원인 확인.
- 무료 CPU 티어는 일정 시간 미사용 시 **sleep** 으로 전환될 수 있다(다음 접속 시
  자동 기동, 첫 응답만 느림). 상시 즉시 응답이 필요하면 유료 하드웨어로 업그레이드.
- gradio 버전 이슈 시 `scripts/deploy_hf.py` 의 `SDK_VERSION` 을 조정해 재배포.
