# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(및 기타 에이전트)를 위한 안내서다.

## 프로젝트 개요

**Synesthetic AI** — 문자-색 공감각(Grapheme-Color Synesthesia)을 NLP로 탐구하는
기말 프로젝트. 핵심 가설은 "텍스트만 학습한 BERT 임베딩 공간에 인간 공감각자와
유사한 색채 구조가 이미 내재되어 있는가?"이며, 이를 텍스트↔이미지 양방향 변환의
예술적 도구(Gradio 데모)로 구현·검증한다.

- 언어/주석: **한국어** (docstring·주석 모두 한국어 기조, 함수명은 영어).
- 저장소 루트는 `synesthesia-nlp/`. 상위 디렉터리는 git 저장소가 아니다.

## 팀 / 담당 영역

| 파트 | 담당 | 파일 |
|---|---|---|
| A 정방향 | 강유영 | `src/forward_pipeline.py`, `src/train_mlp.py` |
| B 역방향 | 이예원 | `src/reverse_pipeline.py` |
| C 시각화·UI | 김민서 | `src/visualizer.py`, `src/app.py` |
| 평가 | 공동 | `src/evaluate_human_vs_ai.py` |

## 실행 방법

```bash
pip install -r requirements.txt          # torch, transformers, numpy, pandas, pillow, plotly, gradio

python src/app.py                        # Gradio 웹 데모 (3개 탭)
python src/train_mlp.py                  # MLP 재학습 → data/mlp_weights.pt 생성
python src/evaluate_human_vs_ai.py       # 보고서용 정량 지표 산출
python src/forward_pipeline.py           # 정방향 자체 검증(assert) 실행
```

> `src/` 모듈은 서로를 **경로 없이** import 한다(`from forward_pipeline import ...`).
> 따라서 항상 `src/` 안에서 실행하거나 `src/`가 `sys.path`에 있어야 한다
> (`app.py`/스크립트들이 entry point).

## 아키텍처

### 정방향 (텍스트 → RGB) — `forward_pipeline.py`
```
텍스트 → BERT 토큰화(서브워드 평균 풀링) → Mean Centering(anisotropy 보정)
  → ① MLP(768→32→3) = rgb_syn
  → ② 앵커 코사인 보정 = rgb_uni
  → blend: β·rgb_syn + (1−β)·rgb_uni, 그 위에 γ(첫 글자 Eagleman 색) 끌림
  → grain(글자별 Eagleman 색 블렌딩) → RGB_out + char_rgbs
```
- `rgb_uni`는 **코퍼스 통계 보정**을 거친다: `load_cosine_calibration()`이
  `candidate_vectors.npy` 전체에서 채널별 코사인의 평균·표준편차와
  지배 채널 편향(`dominance_bias`)을 추정 → `calibrated_anchor_strengths()`가
  표준화 후 softmax(T=2.0)로 R/G/B 비중 산출. 밝기는 원본 코사인 평균으로 별도 계산.
- 색 출처 배지: `color_source()`(Eagleman/NRC=실측, 그 외=예측), 확신도: `color_confidence()`.

### 역방향 (이미지 → 구조적 시) — `reverse_pipeline.py`
```
이미지 → 다운샘플(H×W) → 픽셀 RGB를 앵커 가중합으로 768차원 역변환
  → (coherence>0이면 공간 블러 + 이웃 단어 의미 전파)
  → 후보 9,770단어와 코사인 → 상위 top_k softmax 샘플링 → 격자 시
```
- `_sample_idx()`: 전체 softmax가 균등 랜덤이 되는 문제를 막기 위해 **top_k(기본 40)**
  로 좁힌 뒤 온도 샘플링. coherence가 높을수록 top_k↓·온도↓로 더 결속된 시.

### 시각화 — `visualizer.py`
- `make_color_bar`, `make_2d_image`(글자/단어 단위 토글, N×N 정사각형),
  `make_3d_tower`(Plotly 슬라이더), `make_semantic_space`(PCA 3D, 단어를 `rgb_out`로 채색),
  `make_word_info_panel`(출처 배지 + 확신도 막대).

### UI — `app.py`
- Gradio 3탭: ①정방향 ②역방향 ③순환 실험(텍스트→색→시).
- **성능 핵심**: 정방향은 최초 1회만 BERT 추론 후 `gr.State`에 캐싱,
  β/γ/grain 슬라이더 변경은 `reblend_forward_results()`로 BERT 재추론 없이 즉시 재혼합.

## 데이터 파일 (`data/`)

| 파일 | git | 용도 |
|---|---|---|
| `bert_mean_vec.npy`, `bert_anchor_R/G/B.npy` | ✅ 추적 | Mean Centering 벡터 + RGB 앵커(**mean-centered**) |
| `synesthesia_grapheme_mean_rgb.csv` | ✅ | Eagleman 알파벳-RGB ground truth(26자) |
| `nrc_word_rgb.csv` | ✅ | NRC 단어-RGB 학습쌍(11,449) |
| `poetry_candidate_words.txt` | ✅ | 역방향 후보 단어 **9,770개**(고유명사·인명 제외됨) |
| `candidate_vectors.npy` | ❌ gitignore | 후보 단어 BERT 벡터 캐시(9,770×768). 역방향·코사인 보정·추천에 사용 |
| `mlp_weights.pt` | ❌ gitignore(`*.pt`) | 학습된 MLP. 없으면 rgb_syn→rgb_uni로 graceful fallback |
| `training_vectors*.npy` | ❌ gitignore | train_mlp 중간 캐시 |

- **앵커는 반드시 mean-centered**. `_assert_anchors_centered()`가 앵커 간 cos>0.8이면
  즉시 에러(raw 앵커 오로드 방지). 검증된 centered 앵커는 cos 0.23~0.47.
- `candidate_vectors.npy`가 없으면: 코사인 보정은 `_COSINE_FALLBACK_*` 상수로,
  추천 단어는 큐레이션 폴백으로 동작. 단, **역방향은 9,770개를 즉석 BERT 계산**하므로
  매우 느려진다 → 배포·시연 시 이 캐시 파일을 반드시 동반할 것.

## 주의사항 / 컨벤션

- `load_*` 함수들은 `@cache`로 프로세스당 1회만 로드. BERT 재로딩 비용 없음.
- 0~255 변환은 truncation 대신 `np.rint`(반올림)로 체계적 편향 제거.
- 문서 `docs/명세서.md`는 **원래 명세**, `docs/구현_디벨롭_노트.md`는 명세 대비
  **의도된 발전 사항**을 기록. 구현이 명세와 다르면 디벨롭 노트를 우선 참조/갱신.
- 데이터 수 등 사실이 바뀌면 코드 docstring·README·가이드 문서의 수치도 함께 맞출 것.
