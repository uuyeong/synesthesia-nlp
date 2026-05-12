# 데이터 다운로드 가이드

이 프로젝트는 일부 대용량/원본 데이터를 `.gitignore`로 처리합니다.  
아래 지침에 따라 각자 로컬에서 다운로드 후 지정 경로에 배치하세요.

---

## 포함된 데이터 (GitHub에 있음, 바로 사용 가능)

| 파일 | 경로 | 설명 |
|---|---|---|
| `bert_mean_vec.npy` | `data/` | BERT Mean Centering 평균 벡터 (768차원) |
| `bert_anchor_R/G/B.npy` | `data/` | RGB 앵커 벡터 (768차원 × 3) |
| `nrc_word_rgb.csv` | `data/` | NRC 단어→RGB 변환 완료 (11,449쌍) |
| `synesthesia_grapheme_mean_rgb.csv` | `data/` | Eagleman 알파벳 공감각 데이터 (26쌍) |
| `poetry_candidate_words.txt` | `data/` | 역방향 후보 단어 집합 (9,942개) |

---

## 직접 다운로드가 필요한 데이터

### 1. NRC Word-Colour Association Lexicon (원본)

**파일명:** `NRC-color-lexicon.txt`  
**배치 경로:** `data/NRC-color-lexicon.txt`

**다운로드 방법:**

```bash
# GitHub 미러에서 직접 다운로드
curl -L "https://raw.githubusercontent.com/beefoo/text-analysis/master/color/NRC-color-lexicon.txt" \
     -o data/NRC-color-lexicon.txt
```

> 이미 `nrc_word_rgb.csv`(전처리 완료본)가 포함되어 있으므로,  
> 원본이 필요하지 않으면 생략해도 됩니다.

---

### 2. Eagleman Synesthesia 원본 MATLAB 데이터

**파일명:** `Synesthesia/` 폴더 전체  
**배치 경로:** `data/Synesthesia/`

**다운로드 방법:**

```bash
cd data/
git clone https://github.com/eagleman25/Synesthesia.git
```

> 이미 `synesthesia_grapheme_mean_rgb.csv`(전처리 완료본)가 포함되어 있으므로,  
> `.mat` 파일을 직접 파싱할 필요가 없으면 생략해도 됩니다.

---

### 3. Gutenberg Poetry Corpus (대용량, 학습 불필요 시 생략 가능)

**용도:** BERT Mean Centering `mean_vec` 계산 재현, 후보 단어 집합 재생성  
**배치 경로:** 별도 경로 지정 불필요 (HuggingFace 스트리밍 사용)

**다운로드 방법 (Python):**

```python
from datasets import load_dataset
ds = load_dataset("biglam/gutenberg-poetry-corpus", split="train")
```

> `bert_mean_vec.npy`와 `poetry_candidate_words.txt`가 이미 제공되므로  
> 일반적인 실행 시 코퍼스 재다운로드는 불필요합니다.

---

## 팀원 간 직접 공유 파일 (GitHub 미포함)

### 4. mlp_weights.pt — A(강유영)가 공유

**파일명:** `mlp_weights.pt`  
**배치 경로:** `data/mlp_weights.pt`

**설명:**  
NRC 단어-색상 데이터 11,449쌍 + Eagleman 알파벳 데이터 26쌍으로 학습한 MLP 가중치 파일.  
768차원 BERT 벡터를 입력받아 RGB 3채널(0~255)을 출력하는 신경망(768→32→3)의 학습 결과물.  
`run_forward()` 호출 시 자동으로 로드되므로 **코드 수정 없이 파일만 배치하면 됩니다.**  
파일이 없으면 `rgb_syn` 값이 `rgb_uni`로 대체되어 MLP 결과를 확인할 수 없습니다.

**배치 방법:** A에게 파일을 받아 `data/` 폴더에 넣기

---

### 5. candidate_vectors.npy — B(이예원)가 공유

**파일명:** `candidate_vectors.npy`  
**배치 경로:** `data/candidate_vectors.npy`

**설명:**  
역방향 파이프라인의 후보 단어 9,942개에 대해 BERT 벡터를 사전 계산한 캐시 파일 (shape: 9942×768).  
최초 계산에 약 15~30분이 소요되므로 B가 한 번만 계산해 팀 전체에 공유합니다.  
`run_reverse()` 호출 시 자동으로 로드되므로 **코드 수정 없이 파일만 배치하면 됩니다.**  
파일이 없으면 `run_reverse()` 첫 실행 시 자동 계산이 시작되어 오랜 시간이 걸립니다.

**배치 방법:** B에게 파일을 받아 `data/` 폴더에 넣기

---

## 최종 data/ 폴더 구조 (전부 배치 시)

```
data/
├── bert_mean_vec.npy                    # ✅ GitHub 포함
├── bert_anchor_R.npy                    # ✅ GitHub 포함
├── bert_anchor_G.npy                    # ✅ GitHub 포함
├── bert_anchor_B.npy                    # ✅ GitHub 포함
├── nrc_word_rgb.csv                     # ✅ GitHub 포함
├── synesthesia_grapheme_mean_rgb.csv    # ✅ GitHub 포함
├── poetry_candidate_words.txt           # ✅ GitHub 포함
├── NRC-color-lexicon.txt                # ⬇️ 직접 다운로드
├── Synesthesia/                         # ⬇️ 직접 다운로드
│   ├── data/
│   │   ├── EaglemanColoredAlphabets.mat
│   │   └── ...
│   └── ...
├── mlp_weights.pt                       # 🤝 A(강유영)에게 받기
└── candidate_vectors.npy                # 🤝 B(이예원)에게 받기
```
