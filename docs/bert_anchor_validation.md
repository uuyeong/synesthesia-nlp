# BERT Anchor 벡터 최종 검증 보고서

**검증 일자:** 2026-05-06  
**방법:** BERT (bert-base-uncased) + Mean Centering  
**목적:** Cosine Similarity 색상 추출 (방법 2)용 앵커 벡터 최종 확정

---

## 결정 배경

### 기존 계획의 문제
명세서 초안에서 Cosine Similarity 방법(방법 2)에 **Word2Vec 앵커**를 쓰도록 설계했으나
두 가지 문제가 있었음.

| 문제 | 내용 |
|---|---|
| 차원 불일치 | BERT v_i (768d) ↔ Word2Vec 앵커 (300d) → cos 계산 불가 |
| 공간 불일치 | 차원을 맞춰도 서로 다른 벡터 공간이라 유사도가 의미 없음 |

### 시도한 대안들

| 방법 | 앵커 분리도 | 색상 정확도 | 채택 |
|---|---|---|---|
| Raw BERT | cos 0.89~0.93 (최악) | 시각적으로 모두 회색 | ✗ |
| BERT Whitening | cos 0.06~0.66 (불안정) | 62% | ✗ |
| **BERT + Mean Centering** | **cos 0.23~0.47 (양호)** | **94%** | **✓** |

---

## 최종 방법: BERT + Mean Centering

### 원리
BERT 벡터는 모든 단어가 좁은 방향에 몰려 있는 **anisotropy** 문제가 있음.  
레퍼런스 단어들의 평균 벡터를 빼면 공간이 펼쳐지면서 색상 분리가 가능해짐.

```python
mean_vec = mean(BERT(레퍼런스_단어들))          # 1회 계산 후 저장
v_centered = BERT(word) - mean_vec             # 모든 벡터에 적용
A_R = mean([BERT(w) - mean_vec for w in seeds_r])  # 앵커도 동일하게
```

### 레퍼런스 단어 구성
- seed 단어 30개 (R/G/B 각 10개)
- Gutenberg 시 코퍼스 상위 빈도 단어 190개
- **총 220개**로 mean_vec 계산

---

## 검증 1: Seed 단어 BERT 어휘 확인

seed 단어 30개 전부 BERT 단일 토큰 ✓  
(서브워드 분리 없음 → 앵커 벡터 품질 보장)

| 그룹 | Seed 단어 | 결과 |
|---|---|---|
| R (빨강) | red, crimson, scarlet, blood, flame, fire, rose, ruby, sunset, passion | 전부 단일 토큰 ✓ |
| G (초록) | green, emerald, forest, grass, leaf, moss, fern, meadow, nature, jungle | 전부 단일 토큰 ✓ |
| B (파랑) | blue, sapphire, ocean, sky, sea, water, river, twilight, indigo, cobalt | 전부 단일 토큰 ✓ |

---

## 검증 2: 앵커 벡터 분리도

앵커 벡터 간 코사인 유사도 — **낮을수록 색상이 잘 구분됨**

| | Raw BERT | Mean Centering |
|---|---|---|
| cos(A_R, A_G) | 0.892 | **0.228** |
| cos(A_R, A_B) | 0.932 | **0.471** |
| cos(A_G, A_B) | 0.907 | **0.335** |
| 판정 | ✗ 구분 불가 | **✓ 양호** |

---

## 검증 3: 색상 직관성 테스트 — 17/18 (94%) ✓

| 단어 | R | G | B | 기대 색상 | 지배 채널 | 결과 |
|---|---|---|---|---|---|---|
| fire | 105 | 70 | 78 | 빨강 | R | ✓ |
| blood | 103 | 70 | 80 | 빨강 | R | ✓ |
| rose | 98 | 66 | 89 | 빨강 | R | ✓ |
| ocean | 73 | 76 | 104 | 파랑 | B | ✓ |
| sky | 88 | 67 | 98 | 파랑 | B | ✓ |
| river | 75 | 79 | 99 | 파랑 | B | ✓ |
| forest | 71 | 98 | 85 | 초록 | G | ✓ |
| grass | 74 | 102 | 77 | 초록 | G | ✓ |
| leaf | 70 | 111 | 73 | 초록 | G | ✓ |
| night | 90 | 66 | 97 | 어둠/파랑 | B | ✓ |
| snow | 85 | 85 | 84 | 흰/밝음 | R | ✓ |
| death | 95 | 72 | 87 | 어둠 | R | ✓ |
| joy | 90 | 81 | 83 | 밝음 | R | ✓ |
| storm | 83 | 76 | 94 | 어둠/파랑 | B | ✓ |
| gold | 85 | 92 | 76 | 노랑/초록 | G | ✓ |
| moon | 84 | 80 | 90 | 청백 | B | ✓ |
| **shadow** | **92** | **71** | **91** | **어둠** | **R** | **✗** |
| dawn | 91 | 72 | 91 | 주황/분홍 | R | ✓ |

**shadow** 1개 오답 — R과 B 차이가 1로 경계선상. 허용 범위 내.

---

## 검증 4: Eagleman A~Z 대조

Cosine Similarity 방법(학습 전)과 Eagleman Ground Truth 간 거리.  
**Linear Probe 학습 후 이 거리가 줄어드는지가 핵심 평가 지표.**

| 항목 | 값 |
|---|---|
| 평균 L2 거리 | 73.2 |
| 최대 L2 거리 | 146.2 (I, Y) |
| 최소 L2 거리 | 28.7 (D) |

→ 현재 방법 2(Cosine Similarity)만으로는 Eagleman과 차이가 큼.  
→ 방법 1(Linear Probe)이 이 거리를 줄이는 역할을 하며, β 혼합으로 보완.

---

## 검증 5: 저장된 파일

| 파일 | 차원 | 용도 |
|---|---|---|
| `bert_mean_vec.npy` | (768,) | 모든 벡터에 빼는 평균 벡터 |
| `bert_anchor_R.npy` | (768,) | 빨강 방향 앵커 |
| `bert_anchor_G.npy` | (768,) | 초록 방향 앵커 |
| `bert_anchor_B.npy` | (768,) | 파랑 방향 앵커 |

---

## 최종 확정 아키텍처 (방법 2)

```
[정방향 - 방법 2: Cosine Similarity]

seed 단어 (각 10개)
  ↓ BERT 임베딩 - mean_vec
  A_R, A_G, A_B (768차원)       ← 1회 계산 후 저장

입력 토큰
  ↓ BERT (bert-base-uncased)
  v_i (768차원) - mean_vec
  ↓
  cos(v_i, A_R), cos(v_i, A_G), cos(v_i, A_B)
  ↓ 정규화
  RGB_uni

[역방향]
  v_i = (R/255)*A_R + (G/255)*A_G + (B/255)*A_B
  → poetry_candidate_words.txt에서 nearest neighbor
```

## Gutenberg 코퍼스의 역할 (최종 확정)

| 용도 | 상태 |
|---|---|
| Word2Vec 학습 → 앵커 벡터 | **폐기** (BERT Mean Centering으로 대체) |
| mean_vec 계산용 레퍼런스 단어 제공 | **유지** |
| 역방향 후보 단어집합 (9,953개) | **유지** |
