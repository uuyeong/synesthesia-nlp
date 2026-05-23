"""
forward_pipeline.py — 정방향 파이프라인 (텍스트 → RGB)
담당: 강유영 (A)

흐름:
    텍스트
    → BERT 토크나이저 (서브워드 평균 풀링)
    → BERT 단독 임베딩 + Mean Centering
    → 방법 1: MLP (768→32→3) → RGB_syn
    → 방법 2: Cosine Similarity (앵커 벡터) → RGB_uni
    → β 혼합 + γ 알파벳 노이즈
    → RGB_out
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── MLP 모델 정의 ────────────────────────────────────────────────────────────

class SynesthesiaMLP(nn.Module):
    """768차원 BERT 벡터 → 3차원 RGB 매핑 MLP."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 255.0


# ─── 모델 로드 ────────────────────────────────────────────────────────────────

def load_bert():
    """BERT 토크나이저와 모델을 로드한다.

    Returns:
        tokenizer: BertTokenizer
        model: BertModel (eval 모드)
    """
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    return tokenizer, model


def load_anchors():
    """저장된 BERT Mean Centering 벡터와 RGB 앵커 벡터를 로드한다.

    Returns:
        mean_vec (np.ndarray): shape (768,)
        A_R (np.ndarray): shape (768,)
        A_G (np.ndarray): shape (768,)
        A_B (np.ndarray): shape (768,)
    """
    mean_vec = np.load(DATA_DIR / "bert_mean_vec.npy")
    A_R      = np.load(DATA_DIR / "bert_anchor_R.npy")
    A_G      = np.load(DATA_DIR / "bert_anchor_G.npy")
    A_B      = np.load(DATA_DIR / "bert_anchor_B.npy")
    return mean_vec, A_R, A_G, A_B


def load_mlp(weights_path=None) -> SynesthesiaMLP:
    """학습된 MLP 가중치를 로드한다.

    Args:
        weights_path: mlp_weights.pt 경로 (None이면 data/ 디렉터리 기본값 사용)

    Returns:
        SynesthesiaMLP (eval 모드)
    """
    if weights_path is None:
        weights_path = DATA_DIR / "mlp_weights.pt"
    mlp = SynesthesiaMLP()
    mlp.load_state_dict(torch.load(weights_path, map_location="cpu"))
    mlp.eval()
    return mlp


def load_eagleman() -> dict:
    """synesthesia_grapheme_mean_rgb.csv에서 알파벳 → RGB 딕셔너리를 생성한다.

    Returns:
        dict: {대문자 알파벳 → np.ndarray (3,)} 형태
    """
    df = pd.read_csv(DATA_DIR / "synesthesia_grapheme_mean_rgb.csv")
    eagleman = {}
    for _, row in df.iterrows():
        letter = str(row["grapheme"]).upper()
        rgb = np.array([row["R_0_255"], row["G_0_255"], row["B_0_255"]], dtype=np.float32)
        eagleman[letter] = rgb
    return eagleman


# ─── 임베딩 ──────────────────────────────────────────────────────────────────

def get_bert_vector(word: str, tokenizer, model, mean_vec: np.ndarray) -> np.ndarray:
    """단어를 단독 입력([CLS] word [SEP])하여 BERT 벡터를 추출하고 Mean Centering을 적용한다.

    서브워드가 여러 개인 경우 평균을 취해 단어당 벡터 1개를 반환한다.

    Args:
        word: 입력 단어
        tokenizer: BertTokenizer
        model: BertModel
        mean_vec: BERT anisotropy 보정용 평균 벡터 (768,)

    Returns:
        np.ndarray: Mean Centering 적용된 벡터 (768,)
    """
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    # [CLS](인덱스 0)와 [SEP](인덱스 -1) 제외, 서브워드 토큰만 추출
    subword_vecs = out.last_hidden_state[0, 1:-1, :].detach().numpy()
    vec = subword_vecs.mean(axis=0)  # 서브워드 여러 개 → 평균 풀링
    return vec - mean_vec


def tokenize_text(text: str, tokenizer=None) -> list[str]:
    """텍스트를 알파벳 단어 단위로 분리한다 (구두점·숫자 제외).

    Args:
        text: 입력 텍스트
        tokenizer: 사용하지 않음 (인터페이스 일관성을 위해 유지)

    Returns:
        list[str]: 소문자 단어 리스트
    """
    words = re.findall(r"[a-zA-Z]+", text)
    return [w.lower() for w in words]


# ─── 색상 계산 ────────────────────────────────────────────────────────────────

def rgb_syn(v_i: np.ndarray, mlp: SynesthesiaMLP) -> np.ndarray:
    """방법 1: MLP (768→32→3)로 RGB_syn을 계산한다.

    Args:
        v_i: Mean Centering 적용된 BERT 벡터 (768,)
        mlp: 학습된 SynesthesiaMLP 모델

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255
    """
    with torch.no_grad():
        t = torch.tensor(v_i, dtype=torch.float32).unsqueeze(0)
        out = mlp(t).squeeze(0).numpy()
    return out.astype(np.float32)


def rgb_uni(v_i: np.ndarray, A_R: np.ndarray, A_G: np.ndarray, A_B: np.ndarray) -> np.ndarray:
    """방법 2: Cosine Similarity 기반 RGB_uni를 계산한다.

    cos(v_i, A_R/G/B) ∈ [-1, 1] → (cos+1)/2 ∈ [0,1] → ×255

    Args:
        v_i: Mean Centering 적용된 BERT 벡터 (768,)
        A_R, A_G, A_B: RGB 앵커 벡터 (768,) 각각

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255
    """
    norm_i = np.linalg.norm(v_i) + 1e-8

    def cosine(anchor):
        return np.dot(v_i, anchor) / (norm_i * (np.linalg.norm(anchor) + 1e-8))

    cos_r = cosine(A_R)
    cos_g = cosine(A_G)
    cos_b = cosine(A_B)

    rgb = np.array([(cos_r + 1) / 2,
                    (cos_g + 1) / 2,
                    (cos_b + 1) / 2], dtype=np.float32) * 255.0
    return rgb


def get_grapheme_color(word: str, eagleman: dict) -> np.ndarray:
    """단어의 첫 글자를 Eagleman 데이터에서 조회하여 RGB_grapheme을 반환한다.

    Args:
        word: 입력 단어
        eagleman: {대문자 알파벳 → RGB np.ndarray} 딕셔너리

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255. 알파벳 이외 문자는 (128,128,128) 반환.
    """
    if not word or not word[0].isalpha():
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    key = word[0].upper()
    return eagleman.get(key, np.array([128.0, 128.0, 128.0], dtype=np.float32)).copy()


def blend(rgb_syn_val: np.ndarray, rgb_uni_val: np.ndarray,
          rgb_grapheme: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """β 혼합과 γ 알파벳 노이즈를 적용하여 최종 RGB_out을 계산한다.

    RGB_final = β * RGB_syn + (1-β) * RGB_uni
    RGB_out   = RGB_final + γ * (RGB_grapheme - RGB_final)

    Args:
        rgb_syn_val: MLP 출력 (3,)
        rgb_uni_val: Cosine Similarity 출력 (3,)
        rgb_grapheme: Eagleman 첫 글자 색상 (3,)
        beta: 공감각 비중 (0~1), 1이면 MLP만 사용
        gamma: 알파벳 노이즈 강도 (0~1)

    Returns:
        np.ndarray: 최종 RGB 값 (3,), 범위 0~255
    """
    rgb_final = beta * rgb_syn_val + (1 - beta) * rgb_uni_val
    rgb_out = rgb_final + gamma * (rgb_grapheme - rgb_final)
    return np.clip(rgb_out, 0, 255).astype(np.float32)


# ─── 전체 파이프라인 ──────────────────────────────────────────────────────────

def run_forward(text: str, beta: float = 0.5, gamma: float = 0.0) -> list[dict]:
    """텍스트를 받아 단어별 RGB 색상 리스트를 반환하는 메인 함수.

    Args:
        text: 입력 텍스트 (시, 문장 등)
        beta: 공감각 비중 (0~1)
        gamma: 알파벳 노이즈 강도 (0~1)

    Returns:
        list[dict]: 단어별 결과
            [{"word": str, "rgb_syn": [R,G,B], "rgb_uni": [R,G,B], "rgb_out": [R,G,B]}, ...]
    """
    tokenizer, bert_model = load_bert()
    mean_vec, A_R, A_G, A_B = load_anchors()
    eagleman = load_eagleman()

    mlp_path = DATA_DIR / "mlp_weights.pt"
    if mlp_path.exists():
        mlp = load_mlp(mlp_path)
    else:
        print("[경고] mlp_weights.pt 없음 — rgb_syn을 rgb_uni로 대체합니다.")
        mlp = None

    words = tokenize_text(text)
    results = []

    for word in words:
        v_i = get_bert_vector(word, tokenizer, bert_model, mean_vec)

        r_uni = rgb_uni(v_i, A_R, A_G, A_B)
        r_syn = rgb_syn(v_i, mlp) if mlp is not None else r_uni.copy()
        r_grapheme = get_grapheme_color(word, eagleman)
        r_out = blend(r_syn, r_uni, r_grapheme, beta, gamma)

        results.append({
            "word":    word,
            "rgb_syn": r_syn.astype(int).tolist(),
            "rgb_uni": r_uni.astype(int).tolist(),
            "rgb_out": r_out.astype(int).tolist(),
        })

    return results


# ─── 검증 (python src/forward_pipeline.py 로 실행) ───────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("1단계 검증: load_bert / load_anchors / get_bert_vector")
    print("=" * 60)

    print("\n[1] load_anchors() ...")
    mean_vec, A_R, A_G, A_B = load_anchors()
    assert mean_vec.shape == (768,), f"mean_vec shape 오류: {mean_vec.shape}"
    assert A_R.shape == (768,),      f"A_R shape 오류: {A_R.shape}"
    assert A_G.shape == (768,),      f"A_G shape 오류: {A_G.shape}"
    assert A_B.shape == (768,),      f"A_B shape 오류: {A_B.shape}"
    print("  ✓ mean_vec, A_R, A_G, A_B 모두 shape (768,) 확인")

    print("\n[2] load_bert() ... (처음 실행 시 모델 다운로드로 수분 소요)")
    tokenizer, model = load_bert()
    print("  ✓ BertTokenizer, BertModel 로드 완료")

    print("\n[3] get_bert_vector() 단일 토큰 단어 테스트 ...")
    test_words = ["fire", "ocean", "forest", "night", "snow"]
    for word in test_words:
        vec = get_bert_vector(word, tokenizer, model, mean_vec)
        assert vec.shape == (768,), f"{word} shape 오류: {vec.shape}"
        print(f"  ✓ '{word}' → shape {vec.shape}, norm={np.linalg.norm(vec):.3f}")

    print("\n[4] get_bert_vector() 서브워드 분리 단어 테스트 ...")
    subword_words = ["moonlight", "crimson", "synesthesia"]
    for word in subword_words:
        tokens = tokenizer.tokenize(word)
        vec = get_bert_vector(word, tokenizer, model, mean_vec)
        assert vec.shape == (768,), f"{word} shape 오류: {vec.shape}"
        print(f"  ✓ '{word}' → 서브워드 {tokens} → 평균 풀링 → shape {vec.shape}")

    print("\n[5] 앵커 벡터 코사인 유사도 확인 ...")
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    v_fire   = get_bert_vector("fire",   tokenizer, model, mean_vec)
    v_ocean  = get_bert_vector("ocean",  tokenizer, model, mean_vec)
    v_forest = get_bert_vector("forest", tokenizer, model, mean_vec)

    print(f"  fire  → cos(A_R)={cosine(v_fire, A_R):.3f}  cos(A_G)={cosine(v_fire, A_G):.3f}  cos(A_B)={cosine(v_fire, A_B):.3f}  (기대: R 최대)")
    print(f"  ocean → cos(A_R)={cosine(v_ocean, A_R):.3f}  cos(A_G)={cosine(v_ocean, A_G):.3f}  cos(A_B)={cosine(v_ocean, A_B):.3f}  (기대: B 최대)")
    print(f"  forest→ cos(A_R)={cosine(v_forest, A_R):.3f}  cos(A_G)={cosine(v_forest, A_G):.3f}  cos(A_B)={cosine(v_forest, A_B):.3f}  (기대: G 최대)")

    print("\n" + "=" * 60)
    print("2단계 검증: rgb_uni / load_eagleman / get_grapheme_color / blend / tokenize_text")
    print("=" * 60)

    print("\n[6] rgb_uni() 테스트 ...")
    for word, expected_ch in [("fire", "R"), ("ocean", "B"), ("forest", "G")]:
        v = get_bert_vector(word, tokenizer, model, mean_vec)
        c = rgb_uni(v, A_R, A_G, A_B)
        assert c.shape == (3,), f"rgb_uni shape 오류: {c.shape}"
        assert np.all((c >= 0) & (c <= 255)), f"rgb_uni 범위 오류: {c}"
        print(f"  ✓ '{word}' → rgb_uni={c.astype(int).tolist()}  (기대: {expected_ch} 채널 최대)")

    print("\n[7] load_eagleman() 테스트 ...")
    eagleman = load_eagleman()
    assert "A" in eagleman and eagleman["A"].shape == (3,)
    assert len(eagleman) == 26
    print(f"  ✓ Eagleman 딕셔너리 {len(eagleman)}개 로드 완료")
    print(f"  ✓ A={eagleman['A'].astype(int).tolist()}, R={eagleman['R'].astype(int).tolist()}")

    print("\n[8] get_grapheme_color() 테스트 ...")
    gc = get_grapheme_color("night", eagleman)
    assert gc.shape == (3,)
    assert np.array_equal(gc, eagleman["N"]), "첫 글자 조회 오류"
    gc_num = get_grapheme_color("123", eagleman)
    assert np.array_equal(gc_num, [128, 128, 128]), "숫자 처리 오류"
    print(f"  ✓ 'night' → get_grapheme_color={gc.astype(int).tolist()}  (첫 글자 N)")
    print(f"  ✓ '123'   → {gc_num.astype(int).tolist()}  (알파벳 이외 → 회색)")

    print("\n[9] blend() 테스트 ...")
    r_syn_t = np.array([200.0, 50.0, 100.0])
    r_uni_t = np.array([100.0, 150.0, 80.0])
    r_gra_t = np.array([255.0, 0.0, 0.0])
    r_out = blend(r_syn_t, r_uni_t, r_gra_t, beta=0.5, gamma=0.0)
    expected = np.clip((r_syn_t + r_uni_t) / 2, 0, 255)
    assert np.allclose(r_out, expected), f"blend 오류: {r_out} != {expected}"
    print(f"  ✓ beta=0.5, gamma=0.0 → {r_out.astype(int).tolist()}")

    print("\n[10] tokenize_text() 테스트 ...")
    tokens = tokenize_text("The night was silent and cold.")
    assert tokens == ["the", "night", "was", "silent", "and", "cold"]
    print(f"  ✓ {tokens}")

    print("\n[11] run_forward() 통합 테스트 ...")
    results = run_forward("The night was silent and cold", beta=0.5, gamma=0.0)
    assert len(results) == 6
    for r in results:
        assert set(r.keys()) == {"word", "rgb_syn", "rgb_uni", "rgb_out"}
        assert len(r["rgb_out"]) == 3
        assert all(0 <= v <= 255 for v in r["rgb_out"])
        print(f"  ✓ '{r['word']}' → rgb_out={r['rgb_out']}")

    print("\n" + "=" * 60)
    print("✅ 2단계 검증 완료 — B, C에게 공유 가능")
    print("=" * 60)
