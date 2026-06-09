"""
evaluate_human_vs_ai.py — 인간 공감각자 vs AI 색채 정량 비교 (보고서용 수치 생성)

핵심 질문: "텍스트만 학습한 언어 모델의 임베딩 공간에 인간 공감각자와 유사한
색채 구조가 실제로 내재되어 있는가?" 를 수치로 검증한다.

생성 지표:
  1. 앵커 분리도 — Raw BERT vs Mean-Centered (anisotropy 보정 효과)
  2. 인간 공감각자(Eagleman 26자) vs AI 지배 채널 일치도 + 셔플 베이스라인
  3. NRC 단어 색상 연상 vs AI 지배 채널 일치도 (cosine = 비지도 probe)
  4. 색차 ΔE (CIE76, Lab 공간) — 셔플 베이스라인 대비

주의: Cosine(rgb_uni)은 라벨 없이 앵커만 쓰는 비지도 방식이므로 Eagleman/NRC
전체가 사실상 test 셋이다(공정한 intrinsic-structure 증거). MLP(rgb_syn)는
NRC+Eagleman으로 학습되어 in-sample이므로 참고용으로만 표시한다.

실행:
    python src/evaluate_human_vs_ai.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from forward_pipeline import (
    DATA_DIR, load_bert, load_anchors, get_bert_vector,
    load_eagleman, load_mlp, rgb_syn,
)

# 명세서에 정의된 앵커 seed 단어 (raw vs centered 분리도 재현용)
SEEDS = {
    "R": ["red", "crimson", "scarlet", "blood", "flame",
          "fire", "rose", "ruby", "sunset", "passion"],
    "G": ["green", "emerald", "forest", "grass", "leaf",
          "moss", "fern", "meadow", "nature", "jungle"],
    "B": ["blue", "sapphire", "ocean", "sky", "sea",
          "water", "river", "twilight", "indigo", "cobalt"],
}

SEED = 42


# ─── 색 공간 변환 ──────────────────────────────────────────────────────────────

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0~255) → CIELAB (D65). 입력 shape (..., 3)."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    mask = arr > 0.04045
    arr = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = arr @ M.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e76(rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
    """CIE76 색차 (Lab 유클리드 거리)."""
    return np.linalg.norm(rgb_to_lab(rgb1) - rgb_to_lab(rgb2), axis=-1)


# ─── AI 색 예측 (비지도 cosine, 비증폭 버전) ─────────────────────────────────────

def cosine_rgb_scientific(v_i, A_R, A_G, A_B) -> np.ndarray:
    """과학적 비교용 cosine→RGB (UI의 ×6 채도 증폭 없이 (cos+1)/2 매핑).

    채널 순서(지배 채널)는 증폭 여부와 무관하게 보존되므로 채널 일치도엔 영향이
    없고, ΔE 비교를 공정하게 하기 위해 증폭을 제거한 순수 매핑을 쓴다.
    """
    norm_i = np.linalg.norm(v_i) + 1e-8

    def cos(anchor):
        return float(np.dot(v_i, anchor) / (norm_i * (np.linalg.norm(anchor) + 1e-8)))

    return np.array([(cos(A_R) + 1) / 2, (cos(A_G) + 1) / 2, (cos(A_B) + 1) / 2]) * 255.0


# ─── 지표 계산 ────────────────────────────────────────────────────────────────

def dominant_channel(rgb: np.ndarray) -> np.ndarray:
    """RGB 배열 (N,3) → 채널별 argmax 인덱스 (N,)."""
    return np.argmax(rgb, axis=-1)


def channel_accuracy(pred_rgb: np.ndarray, true_rgb: np.ndarray) -> float:
    return float(np.mean(dominant_channel(pred_rgb) == dominant_channel(true_rgb)))


def rgb_direction_cos(pred_rgb: np.ndarray, true_rgb: np.ndarray) -> float:
    """예측 RGB와 인간 RGB의 방향 일치(중심 제거 후 코사인) 평균.

    밝기/스케일 차이를 빼고 '색의 방향'이 얼마나 일치하는지 보는 연속 지표.
    """
    p = pred_rgb - pred_rgb.mean(axis=1, keepdims=True)
    t = true_rgb - true_rgb.mean(axis=1, keepdims=True)
    pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(pn * tn, axis=1)))


def saliency(rgb: np.ndarray) -> np.ndarray:
    """색 선명도 = max(RGB) - min(RGB). 클수록 뚜렷한 유채색, 작을수록 무채색."""
    return rgb.max(axis=1) - rgb.min(axis=1)


def shuffle_baseline_acc(pred_rgb, true_rgb, rng, trials=1000) -> float:
    """예측을 무작위로 섞었을 때 기대 채널 일치도 (정렬이 없을 때의 베이스라인)."""
    pred_ch = dominant_channel(pred_rgb)
    true_ch = dominant_channel(true_rgb)
    accs = []
    for _ in range(trials):
        accs.append(np.mean(pred_ch[rng.permutation(len(pred_ch))] == true_ch))
    return float(np.mean(accs))


def shuffle_baseline_de(pred_rgb, true_rgb, rng, trials=1000) -> float:
    """예측을 무작위로 섞었을 때 기대 ΔE (정렬이 없을 때의 베이스라인)."""
    des = []
    n = len(pred_rgb)
    for _ in range(trials):
        des.append(np.mean(delta_e76(pred_rgb[rng.permutation(n)], true_rgb)))
    return float(np.mean(des))


# ─── 실험 1: 앵커 분리도 ────────────────────────────────────────────────────────

def eval_anchor_separation(tokenizer, model, mean_vec):
    print("\n" + "=" * 64)
    print("[실험 1] 앵커 분리도 — Raw BERT vs Mean-Centered")
    print("=" * 64)

    raw_anchors, cen_anchors = {}, {}
    for ch, words in SEEDS.items():
        # get_bert_vector는 centering 적용본 → mean_vec 더해 raw 복원
        cen = np.mean([get_bert_vector(w, tokenizer, model, mean_vec) for w in words], axis=0)
        cen_anchors[ch] = cen
        raw_anchors[ch] = cen + mean_vec

    def pair_cos(anchors):
        def c(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return {
            "R-G": c(anchors["R"], anchors["G"]),
            "R-B": c(anchors["R"], anchors["B"]),
            "G-B": c(anchors["G"], anchors["B"]),
        }

    raw_c, cen_c = pair_cos(raw_anchors), pair_cos(cen_anchors)
    print(f"  Raw      앵커간 cos: R-G={raw_c['R-G']:.3f}  R-B={raw_c['R-B']:.3f}  G-B={raw_c['G-B']:.3f}")
    print(f"  Centered 앵커간 cos: R-G={cen_c['R-G']:.3f}  R-B={cen_c['R-B']:.3f}  G-B={cen_c['G-B']:.3f}")
    print(f"  → Raw 평균 {np.mean(list(raw_c.values())):.3f} → Centered 평균 {np.mean(list(cen_c.values())):.3f}")
    return raw_c, cen_c


# ─── 실험 2: Eagleman (인간 공감각자) vs AI ───────────────────────────────────

def eval_eagleman(tokenizer, model, mean_vec, A_R, A_G, A_B, mlp, rng):
    print("\n" + "=" * 64)
    print("[실험 2] 인간 공감각자(Eagleman 26자) vs AI")
    print("=" * 64)

    eagleman = load_eagleman()
    letters = sorted(eagleman.keys())
    human = np.array([eagleman[l] for l in letters])

    cos_pred, mlp_pred = [], []
    for l in letters:
        v = get_bert_vector(l, tokenizer, model, mean_vec)
        cos_pred.append(cosine_rgb_scientific(v, A_R, A_G, A_B))
        if mlp is not None:
            mlp_pred.append(rgb_syn(v, mlp))
    cos_pred = np.array(cos_pred)

    acc = channel_accuracy(cos_pred, human)
    base_acc = shuffle_baseline_acc(cos_pred, human, rng)
    de = float(np.mean(delta_e76(cos_pred, human)))
    base_de = shuffle_baseline_de(cos_pred, human, rng)
    dircos = rgb_direction_cos(cos_pred, human)

    print(f"  [Cosine·비지도] 지배채널 일치 {acc*100:.1f}%  (셔플 베이스라인 {base_acc*100:.1f}%, 우연 33.3%)")
    print(f"  [Cosine·비지도] 평균 ΔE {de:.1f}  (셔플 베이스라인 {base_de:.1f})")
    print(f"  [Cosine·비지도] 색 방향 코사인 {dircos:+.3f}  (1=완전일치, 0=무관)")

    if mlp is not None:
        mlp_pred = np.array(mlp_pred)
        macc = channel_accuracy(mlp_pred, human)
        mde = float(np.mean(delta_e76(mlp_pred, human)))
        print(f"  [MLP·in-sample 참고] 지배채널 일치 {macc*100:.1f}%  평균 ΔE {mde:.1f}")

    # 인간/AI의 지배채널 분포 — 왜 일치가 낮은지 진단
    def dist(rgb):
        ch = dominant_channel(rgb)
        return {c: int(np.sum(ch == i)) for i, c in enumerate("RGB")}
    print(f"  인간 지배채널 분포:  {dist(human)}")
    print(f"  Cosine 지배채널 분포: {dist(cos_pred)}")

    # 예시 5자 (인간 vs AI)
    print("  예시(글자: 인간RGB / AI-Cosine RGB):")
    for l in letters[:6]:
        i = letters.index(l)
        h = human[i].astype(int).tolist()
        c = cos_pred[i].astype(int).tolist()
        print(f"    {l}: {h} / {c}")

    return {"acc": acc, "base_acc": base_acc, "de": de, "base_de": base_de,
            "dircos": dircos, "letters": letters, "human": human, "cos_pred": cos_pred}


# ─── 실험 3: NRC 단어 vs AI ──────────────────────────────────────────────────

def eval_nrc(tokenizer, model, mean_vec, A_R, A_G, A_B, rng, n_sample=300):
    print("\n" + "=" * 64)
    print(f"[실험 3] NRC 단어 색상 연상 vs AI (무작위 {n_sample}개 표본)")
    print("=" * 64)

    df = pd.read_csv(DATA_DIR / "nrc_word_rgb.csv")
    idx = rng.permutation(len(df))[:n_sample]
    sample = df.iloc[idx]

    human, cos_pred, used = [], [], 0
    for _, row in sample.iterrows():
        word = str(row["word"])
        try:
            v = get_bert_vector(word, tokenizer, model, mean_vec)
        except Exception:
            continue
        cos_pred.append(cosine_rgb_scientific(v, A_R, A_G, A_B))
        human.append([row["R"], row["G"], row["B"]])
        used += 1

    human, cos_pred = np.array(human), np.array(cos_pred)
    acc = channel_accuracy(cos_pred, human)
    base_acc = shuffle_baseline_acc(cos_pred, human, rng)
    de = float(np.mean(delta_e76(cos_pred, human)))
    base_de = shuffle_baseline_de(cos_pred, human, rng)

    print(f"  표본 {used}개 (전체)")
    print(f"  [전체] 지배채널 일치 {acc*100:.1f}%  (셔플 {base_acc*100:.1f}%, 우연 33.3%)  ΔE {de:.1f}")

    # 색 선명도(saliency)로 분리: 인간이 뚜렷한 색을 부여한 단어 vs 무채색/모호 단어
    sal = saliency(human)
    thr = 60.0  # max-min >= 60 → 뚜렷한 유채색으로 간주
    vivid = sal >= thr
    muted = ~vivid
    if vivid.sum() > 0:
        v_acc = channel_accuracy(cos_pred[vivid], human[vivid])
        v_base = shuffle_baseline_acc(cos_pred[vivid], human[vivid], rng)
        print(f"  [뚜렷한 색 단어 {int(vivid.sum())}개] 지배채널 일치 {v_acc*100:.1f}%  (셔플 {v_base*100:.1f}%)")
    if muted.sum() > 0:
        m_acc = channel_accuracy(cos_pred[muted], human[muted])
        print(f"  [무채색·모호 단어 {int(muted.sum())}개] 지배채널 일치 {m_acc*100:.1f}%")

    return {"n": used, "acc": acc, "base_acc": base_acc, "de": de, "base_de": base_de,
            "v_acc": v_acc if vivid.sum() else None, "n_vivid": int(vivid.sum())}


def eval_curated(tokenizer, model, mean_vec, A_R, A_G, A_B):
    """명백히 색을 띤 단어 집합 — '94% 직관성' 주장 재현/검증."""
    print("\n" + "=" * 64)
    print("[실험 4] 색을 띤 비-seed 단어 (일반화·직관성 검증)")
    print("=" * 64)
    # seed 단어(red/fire/ocean/forest 등)와 겹치지 않는 단어만 사용해 순환 방지.
    # (단어, 기대 지배채널)
    curated = [
        ("cherry", "R"), ("brick", "R"), ("tomato", "R"), ("coral", "R"),
        ("navy", "B"), ("azure", "B"), ("teal", "B"), ("denim", "B"),
        ("lime", "G"), ("mint", "G"), ("olive", "G"), ("jade", "G"),
    ]
    ch_idx = {"R": 0, "G": 1, "B": 2}
    hit = 0
    for word, exp in curated:
        v = get_bert_vector(word, tokenizer, model, mean_vec)
        pred = cosine_rgb_scientific(v, A_R, A_G, A_B)
        got = "RGB"[int(np.argmax(pred))]
        ok = (got == exp)
        hit += ok
        print(f"    {word:8s} 기대 {exp} → 예측 {got} {'✓' if ok else '✗'}  {pred.astype(int).tolist()}")
    print(f"  직관성 정확도: {hit}/{len(curated)} ({hit/len(curated)*100:.0f}%)")
    return {"hit": hit, "total": len(curated)}


def main():
    rng = np.random.default_rng(SEED)
    print("BERT / 앵커 로드 중 ...")
    tokenizer, model = load_bert()
    mean_vec, A_R, A_G, A_B = load_anchors()
    try:
        mlp = load_mlp()
    except FileNotFoundError:
        mlp = None
        print("  (mlp_weights.pt 없음 — MLP 참고 지표 생략)")

    eval_anchor_separation(tokenizer, model, mean_vec)
    eval_eagleman(tokenizer, model, mean_vec, A_R, A_G, A_B, mlp, rng)
    eval_nrc(tokenizer, model, mean_vec, A_R, A_G, A_B, rng)
    eval_curated(tokenizer, model, mean_vec, A_R, A_G, A_B)
    print("\n완료. 위 수치를 docs/인간_vs_AI_색채_비교.md 에 반영하세요.")


if __name__ == "__main__":
    main()
