# synesthesia-nlp

**단어 벡터를 통한 문자-색 공감각의 시각화**  
Visualization of Grapheme-Color Synesthesia through Word Vectors

---

## 프로젝트 개요

문자-색 공감각(Grapheme-Color Synesthesia)은 글자나 단어를 볼 때 특정 색상이 자동으로 연상되는 인지 현상입니다.  
BERT 임베딩 공간에 이미 색채 구조가 내재되어 있는지 탐구하고, 이를 텍스트↔이미지 양방향 변환의 예술적 도구로 활용합니다.

---

## 팀 구성

| 역할 | 담당자 | 파트 |
|---|---|---|
| A | 강유영 | 정방향 파이프라인 (`src/forward_pipeline.py`) |
| B | 이예원 | 역방향 파이프라인 (`src/reverse_pipeline.py`) |
| C | 김민서 | 3D 시각화 & 데모 UI (`src/visualizer.py`, `src/app.py`) |

---

## 폴더 구조

```
synesthesia-nlp/
├── src/
│   ├── forward_pipeline.py   # 텍스트 → RGB (강유영)
│   ├── reverse_pipeline.py   # 이미지 → 시 (이예원)
│   ├── visualizer.py         # 2D/3D 시각화 (김민서)
│   └── app.py                # Gradio 데모 (김민서)
├── data/
│   ├── bert_mean_vec.npy               # BERT Mean Centering 벡터
│   ├── bert_anchor_R/G/B.npy           # RGB 앵커 벡터
│   ├── nrc_word_rgb.csv                # NRC 단어→RGB (11,449쌍)
│   ├── synesthesia_grapheme_mean_rgb.csv  # Eagleman 데이터 (26쌍)
│   └── poetry_candidate_words.txt      # 역방향 후보 단어 (9,942개)
├── docs/
│   ├── 명세서.md                        # 프로젝트 명세서
│   ├── 교수님_피드백_반영.md             # 교수님 피드백 및 검증 실험
│   ├── bert_anchor_validation.md       # BERT 앵커 벡터 검증 보고서
│   └── data_validation_report.md       # 데이터셋 검증 보고서
├── DATA_GUIDE.md             # 대용량 데이터 다운로드 안내
└── README.md
```

---

## 핵심 아키텍처

```
[정방향] 텍스트 → BERT(Isolated) + Mean Centering → MLP(768→32→3) + Cosine Sim → RGB
[역방향] 이미지 → 다운샘플링 → Anchor 역변환 → 후보 단어 검색 → 구조적 시
```

- **BERT**: bert-base-uncased, 단어 단독 입력(Isolated), 768차원
- **Mean Centering**: BERT anisotropy 보정 (앵커 분리도 0.89→0.23)
- **MLP**: 768→32→3, ReLU + Dropout(0.3)
- **Cosine Similarity**: R/G/B 앵커 벡터 기반 색상 계산
- **β 혼합**: `RGB_final = β*RGB_syn + (1-β)*RGB_uni`

---

## 데이터

GitHub에 포함된 소용량 전처리 완료 파일은 `data/` 폴더에 있습니다.  
대용량 원본 데이터 다운로드 방법은 **[DATA_GUIDE.md](DATA_GUIDE.md)** 를 참고하세요.

---

## 실행 (추후 업데이트 예정)

```bash
pip install -r requirements.txt
python src/app.py
```
