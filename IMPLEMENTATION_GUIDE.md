# 구현 가이드 — 팀 협업용

**프로젝트:** synesthesia-nlp  
**작성일:** 2026-05-08  
**팀 구성:**

| 역할 | 담당자 | 파일 |
|---|---|---|
| A | 강유영 | `src/forward_pipeline.py` |
| B | 이예원 | `src/reverse_pipeline.py` |
| C | 김민서 | `src/visualizer.py`, `src/app.py` |

---

## 전체 구현 순서 한눈에 보기

```
1단계 ── A 단독 ──────────────────────────────────────────────────
         BERT 로드 + get_bert_vector() + load_anchors()
         ※ B와 C 모두 이 함수에 의존 → 가장 먼저 완성

2단계 ── A · B · C 병렬 진행 ────────────────────────────────────
         A: MLP 학습 + rgb_syn/uni/blend + run_forward()
         B: 이미지 처리 + 역변환 + 단어 검색 + run_reverse()
         C: 2D/3D 시각화 함수 + Gradio UI 레이아웃

3단계 ── C 통합 ──────────────────────────────────────────────────
         A, B의 run_* 함수를 app.py에서 연결
         슬라이더 연동 + 탭 완성 + 순환 실험 시연 구성

4단계 ── 전체 통합 테스트 ────────────────────────────────────────
         "The night was silent and cold" 정방향 테스트
         노을 사진 역방향 테스트
         순환 실험 (정방향 → 역방향 → 비교)
```

---

## 1단계: A 단독 — 공유 기반 함수 완성

> **완료 전까지 B와 C는 본격 구현 불가.** A가 가장 먼저 해야 할 작업.

### A가 구현할 함수

#### `load_bert()` → `(tokenizer, model)`

```python
from transformers import BertTokenizer, BertModel

def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    return tokenizer, model
```

- 반환: `(BertTokenizer, BertModel)`
- B와 C는 이 함수를 `from forward_pipeline import load_bert`로 가져다 씀

---

#### `load_anchors()` → `(mean_vec, A_R, A_G, A_B)`

```python
def load_anchors():
    # data/ 폴더의 .npy 파일 4개 로드
    mean_vec = np.load(DATA_DIR / "bert_mean_vec.npy")
    A_R      = np.load(DATA_DIR / "bert_anchor_R.npy")
    A_G      = np.load(DATA_DIR / "bert_anchor_G.npy")
    A_B      = np.load(DATA_DIR / "bert_anchor_B.npy")
    return mean_vec, A_R, A_G, A_B
```

- 반환: 768차원 np.ndarray 4개
- B가 역변환(`pixel_to_vector`)에 A_R/G/B를 사용
- B가 후보 단어 벡터 계산에 mean_vec을 사용

---

#### `get_bert_vector(word, tokenizer, model, mean_vec)` → `np.ndarray (768,)`

> **이 함수가 1단계의 핵심.** B의 후보 단어 9,942개 사전 계산이 이 함수에 의존.

```python
def get_bert_vector(word, tokenizer, model, mean_vec):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
    # [CLS]=0, [SEP]=-1 제외하고 서브워드 토큰만 추출
    subword_vecs = out.last_hidden_state[0, 1:-1, :].numpy()
    # 서브워드가 여러 개면 평균 풀링 → 단어당 벡터 1개
    vec = subword_vecs.mean(axis=0)
    return vec - mean_vec  # Mean Centering 적용
```

- 입력: 단어 문자열 + BERT 객체들
- 출력: Mean Centering 완료된 768차원 벡터
- **B가 import해서 사용:** 키워드 문맥 벡터, 후보 단어 벡터 사전 계산

---

### 1단계 완료 기준 (A → 팀 공유)

- [ ] `load_bert()` 실행 시 오류 없음
- [ ] `get_bert_vector("fire", ...)` 실행 시 shape `(768,)` 반환 확인
- [ ] `load_anchors()` 실행 시 4개 npy 정상 로드 확인
- [ ] 완료 후 팀 단톡에 공유 → B, C 병렬 시작 신호

---

## 2단계: A · B · C 병렬 진행

### A — 정방향 파이프라인 완성

**구현 순서:**

#### ① MLP 모델 정의 및 학습

```
파일 위치: src/forward_pipeline.py (또는 별도 src/train_mlp.py)

MLP 구조:
    입력 (768) → Linear → ReLU → Dropout(0.3) → Linear → Sigmoid×255 → 출력 (3)
    중간 레이어: 32차원

학습 데이터: data/nrc_word_rgb.csv (11,449쌍) + data/synesthesia_grapheme_mean_rgb.csv (26쌍)
손실 함수: MSE Loss
옵티마이저: Adam, lr=0.001
에폭: 500 이상 (validation loss 기준 early stopping 권장)

저장: data/mlp_weights.pt  ← .gitignore에 등록되어 있음, 로컬에서만 사용
```

#### ② `rgb_syn(v_i, mlp)` → `np.ndarray (3,)`

- MLP forward pass로 RGB_syn 계산
- 출력 범위: 0~255

#### ③ `rgb_uni(v_i, A_R, A_G, A_B)` → `np.ndarray (3,)`

```
cos_r = cosine_similarity(v_i, A_R)
cos_g = cosine_similarity(v_i, A_G)
cos_b = cosine_similarity(v_i, A_B)
→ [0,1] 정규화 후 ×255
```

#### ④ `get_grapheme_color(word, eagleman)` → `np.ndarray (3,)`

- 단어 첫 글자 → Eagleman 딕셔너리 조회
- 알파벳 이외 문자 → `(128, 128, 128)` 반환

#### ⑤ `blend(rgb_syn, rgb_uni, rgb_grapheme, beta, gamma)` → `np.ndarray (3,)`

```
RGB_final = β * RGB_syn + (1-β) * RGB_uni
RGB_out   = RGB_final + γ * (RGB_grapheme - RGB_final)
→ np.clip(result, 0, 255)
```

#### ⑥ `run_forward(text, beta, gamma)` → `list[dict]`

```python
# C가 기대하는 출력 형식 (반드시 이 형식으로 반환)
[
    {
        "word":        "night",
        "rgb_syn":     [90, 66, 97],   # MLP 출력
        "rgb_uni":     [88, 65, 100],  # Cosine Similarity 출력
        "rgb_out":     [89, 65, 98],   # 최종 혼합 결과
    },
    ...
]
```

> ⚠️ **C와 합의된 출력 형식** — 키 이름과 값 범위(0~255 정수)를 반드시 지킬 것

---

### B — 역방향 파이프라인 완성

> 1단계 완료 후 시작. `from forward_pipeline import load_bert, load_anchors, get_bert_vector` 사용.

**구현 순서:**

#### ① `load_candidate_words()` → `list[str]`

- `data/poetry_candidate_words.txt` 읽기
- 개행 기준으로 split → 9,942개 단어 리스트

#### ② `load_candidate_vectors(candidate_words, tokenizer, model, mean_vec)` → `np.ndarray (9942, 768)`

```
캐시 로직:
    data/candidate_vectors.npy 있으면 → np.load()로 바로 사용
    없으면 → get_bert_vector()로 9,942개 계산 후 저장
    (최초 1회만 계산, 이후 수초 내 로드)
```

- 이 파일도 `.gitignore`에 추가 필요 (대용량)
- 최초 계산 시 약 15~30분 소요 예상 → 팀 중 한 명이 계산 후 공유 권장

#### ③ `downsample_image(image_path, resolution)` → `np.ndarray (H, W, 3)`

```python
from PIL import Image
img = Image.open(image_path).resize((W, H), Image.LANCZOS).convert("RGB")
return np.array(img)
```

#### ④ `pixel_to_vector(rgb, A_R, A_G, A_B)` → `np.ndarray (768,)`

```
v_i = (R/255)*A_R + (G/255)*A_G + (B/255)*A_B
```

#### ⑤ `find_nearest_word(v_i, candidate_matrix, candidate_words, v_context, alpha)` → `str`

```
sim_color   = candidate_matrix @ v_i / (||candidate_matrix|| * ||v_i||)  # (N,)
sim_context = candidate_matrix @ v_context / ... (alpha > 0일 때만)

score = alpha * sim_context + (1-alpha) * sim_color
→ argmax(score) 또는 softmax 샘플링으로 단어 선택
```

#### ⑥ `run_reverse(image_path, resolution, keyword, alpha)` → `list[list[str]]`

```python
# C가 기대하는 출력 형식 (반드시 이 형식으로 반환)
[
    ["burn",  "flame", "glow",  "heat"],   # 1행
    ["ember", "dusk",  "fade",  "amber"],  # 2행
    ...
]
# resolution=(H, W)이면 H개 행, 각 행 W개 단어
```

> ⚠️ **C와 합의된 출력 형식** — 반드시 `list[list[str]]` 형태, H×W 보장

---

### C — 시각화 및 UI 레이아웃

> A의 `run_forward()` 출력 형식과 B의 `run_reverse()` 출력 형식을 기준으로 구현.
> A, B 함수가 완성 안 되어 있어도 **더미 데이터로 UI 먼저 구축 가능.**

#### ① `make_color_bar(word_colors, pixel_width, pixel_height)` → `PIL.Image`

- `word_colors[i]["rgb_out"]` 값으로 단어별 색상 블록 생성
- 가로로 이어 붙여서 색상 바 이미지 반환

#### ② `make_2d_image(word_colors, ncols)` → `PIL.Image`

- `rgb_out` 값을 raster scan (행 우선) 으로 배열
- 한 단어 = 한 픽셀, 부족하면 `(128,128,128)`으로 패딩

#### ③ `rgb_to_barycentric(r, g, b)` → `(x, y)`

- RGB → 등변삼각형 무게중심 좌표 변환
- 3D 타워 XY 좌표로 사용

#### ④ `make_3d_tower(word_colors)` → `plotly.Figure`

- XY: barycentric 좌표, Z: 단어 순서
- marker 색상: `rgb_out` 값
- hover: 단어명 + rgb_syn / rgb_uni / rgb_out

#### ⑤ Gradio UI 레이아웃 (`app.py`)

```
더미 데이터로 레이아웃 먼저 완성 가능:
dummy_colors = [{"word": "night", "rgb_syn": [90,66,97],
                 "rgb_uni": [88,65,100], "rgb_out": [89,65,98]}]
→ make_color_bar(dummy_colors), make_2d_image(dummy_colors), make_3d_tower(dummy_colors) 테스트
```

---

## 3단계: C 통합

A와 B의 `run_*` 함수가 완성되면 C가 `app.py`에서 연결.

```python
# app.py
from forward_pipeline import run_forward
from reverse_pipeline import run_reverse
from visualizer import make_color_bar, make_2d_image, make_3d_tower

def forward_tab_handler(text, beta, gamma, ncols):
    word_colors = run_forward(text, beta, gamma)
    return (
        make_color_bar(word_colors),
        make_2d_image(word_colors, ncols=ncols),
        make_3d_tower(word_colors)
    )

def reverse_tab_handler(image, resolution_str, keyword, alpha):
    H, W = map(int, resolution_str.split("×"))
    poem_grid = run_reverse(image, (H, W), keyword or None, alpha)
    return "\n".join(" ".join(row) for row in poem_grid)
```

---

## 4단계: 전체 통합 테스트

### 테스트 1 — 정방향

```
입력: "The night was silent and cold"
beta=0.5, gamma=0.0
기대: night → 어둠/파랑 계열, cold → 파랑 계열
```

### 테스트 2 — 역방향

```
입력: 노을 사진 (붉은/주황 계열)
resolution=8×8, keyword=없음
기대: flame, ember, dusk, glow 같은 따뜻한 계열 단어
```

### 테스트 3 — 순환 실험

```
시 입력 → 정방향 → 색상 이미지 → 역방향 → 새로운 시
원본 시와 순환 후 시를 나란히 비교
```

---

## 인터페이스 합의 요약

> 이 표에 있는 함수 시그니처와 출력 형식은 팀원 간 반드시 지켜야 합니다.  
> 변경 필요 시 단톡에서 합의 후 수정.

| 함수 | 구현 | 사용 | 출력 형식 |
|---|---|---|---|
| `load_bert()` | A | A, B, C | `(tokenizer, model)` |
| `load_anchors()` | A | A, B | `(mean_vec, A_R, A_G, A_B)` — 각 `(768,)` |
| `get_bert_vector(word, ...)` | A | A, B | `np.ndarray (768,)` |
| `run_forward(text, beta, gamma)` | A | C | `list[dict]` — 아래 형식 |
| `run_reverse(image, resolution, keyword, alpha)` | B | C | `list[list[str]]` — H×W |

**`run_forward` 출력 형식:**
```python
[{"word": str, "rgb_syn": [int,int,int], "rgb_uni": [int,int,int], "rgb_out": [int,int,int]}, ...]
```

**`run_reverse` 출력 형식:**
```python
[["word", "word", ...], ["word", "word", ...], ...]  # H개 행, 각 행 W개 단어
```

---

## 의존성 그래프

```
A: load_bert()
A: load_anchors()
A: get_bert_vector()
        │
        ├──► B: load_candidate_vectors()
        │         │
        │         └──► B: find_nearest_word()
        │                       │
        │                       └──► B: run_reverse() ──► C: app.py (탭2)
        │
        └──► A: rgb_syn() / rgb_uni() / blend()
                       │
                       └──► A: run_forward() ──► C: app.py (탭1)

C: make_color_bar()  ┐
C: make_2d_image()   ├──► C: app.py (탭1 출력)
C: make_3d_tower()   ┘
```

---

## 파일 구조 최종 확인

```
synesthesia-nlp/
├── src/
│   ├── forward_pipeline.py   ← A 담당
│   ├── reverse_pipeline.py   ← B 담당
│   ├── visualizer.py         ← C 담당
│   └── app.py                ← C 담당 (진입점)
├── data/
│   ├── bert_mean_vec.npy               ✅ GitHub 포함
│   ├── bert_anchor_R/G/B.npy           ✅ GitHub 포함
│   ├── nrc_word_rgb.csv                ✅ GitHub 포함
│   ├── synesthesia_grapheme_mean_rgb.csv ✅ GitHub 포함
│   ├── poetry_candidate_words.txt      ✅ GitHub 포함
│   ├── mlp_weights.pt                  ⛔ .gitignore (A가 로컬 학습)
│   └── candidate_vectors.npy           ⛔ .gitignore (B가 로컬 계산)
├── docs/
│   ├── 명세서.md
│   ├── 교수님_피드백_반영.md
│   ├── bert_anchor_validation.md
│   └── data_validation_report.md
├── DATA_GUIDE.md
├── IMPLEMENTATION_GUIDE.md   ← 이 파일
└── README.md
```

---

## 추가 주의사항

**mlp_weights.pt 공유 방법**

A가 MLP 학습 완료 후 weights 파일을 팀원들에게 공유해야 B와 C가 테스트 가능합니다.  
GitHub에는 올리지 않으므로 카카오톡 또는 구글 드라이브로 공유.

**candidate_vectors.npy 공유 방법**

B가 9,942개 단어 벡터 계산 완료 후 팀원들에게 공유.  
최초 계산에 시간이 걸리므로 한 명만 계산하고 나머지는 받아서 사용.

**실행 환경**

```bash
pip install transformers torch numpy pandas pillow plotly gradio
```
