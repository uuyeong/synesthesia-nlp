"""
train_mlp.py — MLP 학습 스크립트 (A 파트 2단계)

학습 데이터:
    data/nrc_word_rgb.csv              (11,449쌍)
    data/synesthesia_grapheme_mean_rgb.csv  (26쌍)

실행:
    python src/train_mlp.py

저장:
    data/mlp_weights.pt              (gitignore — 팀원에게 직접 공유)
    data/training_vectors.npy        (gitignore — BERT 벡터 캐시)
    data/training_vectors.valid.npy  (gitignore — 유효 인덱스 마스크)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from forward_pipeline import (
    DATA_DIR, SynesthesiaMLP,
    load_bert, load_anchors, get_bert_vector,
)


def load_training_pairs() -> list[tuple[str, list[float], str]]:
    """NRC 단어-RGB 쌍 + Eagleman 자소-RGB 쌍을 source 라벨과 함께 반환한다.

    Returns:
        list of (word, [R, G, B], source) — source는 "nrc" 또는 "eagleman"
    """
    pairs = []
    nrc_df = pd.read_csv(DATA_DIR / "nrc_word_rgb.csv")
    for _, row in nrc_df.iterrows():
        pairs.append((
            str(row["word"]),
            [float(row["R"]), float(row["G"]), float(row["B"])],
            "nrc",
        ))

    grapheme_df = pd.read_csv(DATA_DIR / "synesthesia_grapheme_mean_rgb.csv")
    for _, row in grapheme_df.iterrows():
        letter = str(row["grapheme"]).lower()
        rgb = [float(row["R_0_255"]), float(row["G_0_255"]), float(row["B_0_255"])]
        pairs.append((letter, rgb, "eagleman"))

    return pairs


def compute_bert_vectors(pairs, tokenizer, model, mean_vec,
                         cache_path: Path):
    """단어별 BERT 벡터를 계산·캐시하고 유효 인덱스 마스크를 반환한다.

    실패한 단어(예: 빈 문자열, [UNK]만 잡힌 경우 등)는 zero 벡터로 남기고
    valid 마스크에서 제외하여 이후 학습에서 사용하지 않는다.
    (이전엔 zero 벡터가 학습 데이터로 들어가 MLP를 오염시킬 수 있었다.)

    Returns:
        X (np.ndarray):     (N, 768) — 모든 단어의 BERT 벡터 (실패는 zeros)
        Y (np.ndarray):     (N, 3)   — 정답 RGB
        sources (np.ndarray): (N,)    — "nrc" / "eagleman" 라벨
        valid (np.ndarray): (N,)    — 유효 인덱스 bool 마스크
    """
    words   = [w for w, _, _ in pairs]
    rgbs    = [rgb for _, rgb, _ in pairs]
    sources = np.array([s for _, _, s in pairs])

    valid_path = cache_path.with_suffix(".valid.npy")

    if cache_path.exists() and valid_path.exists():
        print(f"  캐시 로드: {cache_path}")
        X = np.load(cache_path)
        valid = np.load(valid_path)
    else:
        print(f"  BERT 벡터 계산 중 ({len(words)}개) ...")
        X = np.zeros((len(words), 768), dtype=np.float32)
        valid = np.zeros(len(words), dtype=bool)
        fail_log = []
        for i, word in enumerate(words):
            if i % 500 == 0:
                print(f"    {i}/{len(words)}")
            try:
                X[i] = get_bert_vector(word, tokenizer, model, mean_vec)
                valid[i] = True
            except Exception as e:
                fail_log.append((word, type(e).__name__))
        np.save(cache_path, X)
        np.save(valid_path, valid)
        print(f"  캐시 저장: {cache_path}")
        if fail_log:
            print(f"  [경고] BERT 벡터 계산 실패 {len(fail_log)}개 (학습에서 제외):")
            for w, err in fail_log[:10]:
                print(f"    '{w}' ({err})")
            if len(fail_log) > 10:
                print(f"    ... +{len(fail_log) - 10}개")

    Y = np.array(rgbs, dtype=np.float32)
    return X, Y, sources, valid


def _eval_subset_loss(mlp, criterion, X, Y, mask) -> float:
    """source 라벨 기준 부분집합의 MSE를 계산한다."""
    if not mask.any():
        return float("nan")
    Xs = torch.tensor(X[mask], dtype=torch.float32)
    Ys = torch.tensor(Y[mask], dtype=torch.float32)
    with torch.no_grad():
        return criterion(mlp(Xs), Ys).item()


def train(epochs: int = 500, lr: float = 0.001, patience: int = 30,
          batch_size: int = 256, seed: int = 42) -> None:
    # 결정성 확보 — 학습 후 weights를 팀원과 공유할 때 환경 차이를 줄인다.
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("MLP 학습 시작")
    print("=" * 60)

    print("\n[1] BERT 로드 ...")
    tokenizer, bert_model = load_bert()
    mean_vec, _, _, _ = load_anchors()

    print("\n[2] 학습 데이터 준비 ...")
    pairs = load_training_pairs()
    print(f"  총 {len(pairs)}쌍")

    cache_path = DATA_DIR / "training_vectors.npy"
    X, Y, sources, valid = compute_bert_vectors(
        pairs, tokenizer, bert_model, mean_vec, cache_path
    )

    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"\n  유효 데이터 {valid.sum()}/{len(valid)} (실패 {n_invalid}개 제외)")
    X = X[valid]
    Y = Y[valid]
    sources = sources[valid]

    print(f"\n[3] X shape={X.shape}, Y shape={Y.shape}")
    print(f"  source 분포: nrc={(sources == 'nrc').sum()}, "
          f"eagleman={(sources == 'eagleman').sum()}")

    # Train / Val 분리 (90 / 10) — torch 시드와 별도로 numpy 시드 사용
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X[val_idx],   dtype=torch.float32)
    Y_val   = torch.tensor(Y[val_idx],   dtype=torch.float32)

    print(f"  train={len(train_idx)}, val={len(val_idx)}")

    print("\n[4] 학습 ...")
    mlp = SynesthesiaMLP()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # mini-batch SGD — full-batch보다 일반화 성능이 안정적이고 노이즈 효과로
    # local minima 탈출에 유리하다.
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = mlp(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / len(X_train)

        mlp.eval()
        with torch.no_grad():
            val_loss = criterion(mlp(X_val), Y_val).item()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.2f} | val={val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    mlp.load_state_dict(best_state)
    mlp.eval()
    weights_path = DATA_DIR / "mlp_weights.pt"
    torch.save(mlp.state_dict(), weights_path)

    print(f"\n✅ 학습 완료")
    print(f"   저장: {weights_path}")
    print(f"   Best val MSE = {best_val_loss:.2f}  (RGB 0~255 스케일)")

    # source별 분리 평가 — Eagleman 26쌍이 단순 셔플로 val에 거의 안 들어가서
    # 단일 val_loss로는 알파벳 학습 품질을 알 수 없다.
    nrc_mask = sources == "nrc"
    eagleman_mask = sources == "eagleman"
    nrc_loss = _eval_subset_loss(mlp, criterion, X, Y, nrc_mask)
    eagleman_loss = _eval_subset_loss(mlp, criterion, X, Y, eagleman_mask)
    print(f"   전체 NRC      MSE = {nrc_loss:.2f}  ({nrc_mask.sum()}쌍)")
    print(f"   전체 Eagleman MSE = {eagleman_loss:.2f}  ({eagleman_mask.sum()}쌍)")

    print(f"\n⚠️  mlp_weights.pt를 팀 카카오톡 또는 구글 드라이브로 공유해주세요.")


if __name__ == "__main__":
    train()
