"""
train_mlp.py — MLP 학습 스크립트 (A 파트 2단계)

학습 데이터:
    data/nrc_word_rgb.csv              (11,449쌍)
    data/synesthesia_grapheme_mean_rgb.csv  (26쌍)

실행:
    python src/train_mlp.py

저장:
    data/mlp_weights.pt       (gitignore — 팀원에게 직접 공유)
    data/training_vectors.npy (gitignore — BERT 벡터 캐시)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from forward_pipeline import (
    DATA_DIR, SynesthesiaMLP,
    load_bert, load_anchors, get_bert_vector,
)


def load_training_pairs() -> list[tuple[str, list[float]]]:
    """NRC 단어-RGB 쌍 + Eagleman 자소-RGB 쌍을 반환한다."""
    nrc_df = pd.read_csv(DATA_DIR / "nrc_word_rgb.csv")
    pairs = [(str(row["word"]), [float(row["R"]), float(row["G"]), float(row["B"])])
             for _, row in nrc_df.iterrows()]

    grapheme_df = pd.read_csv(DATA_DIR / "synesthesia_grapheme_mean_rgb.csv")
    for _, row in grapheme_df.iterrows():
        letter = str(row["grapheme"]).lower()
        rgb = [float(row["R_0_255"]), float(row["G_0_255"]), float(row["B_0_255"])]
        pairs.append((letter, rgb))

    return pairs


def compute_bert_vectors(pairs, tokenizer, model, mean_vec,
                         cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """단어별 BERT 벡터를 계산하고 캐시한다 (최초 1회만 계산)."""
    words = [w for w, _ in pairs]
    rgbs  = [rgb for _, rgb in pairs]

    if cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        X = np.load(cache_path)
    else:
        print(f"  BERT 벡터 계산 중 ({len(words)}개) ...")
        X = np.zeros((len(words), 768), dtype=np.float32)
        for i, word in enumerate(words):
            if i % 500 == 0:
                print(f"    {i}/{len(words)}")
            try:
                X[i] = get_bert_vector(word, tokenizer, model, mean_vec)
            except Exception:
                pass  # 계산 실패 시 zero 벡터 유지
        np.save(cache_path, X)
        print(f"  캐시 저장: {cache_path}")

    Y = np.array(rgbs, dtype=np.float32)
    return X, Y


def train(epochs: int = 500, lr: float = 0.001, patience: int = 30) -> None:
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
    X, Y = compute_bert_vectors(pairs, tokenizer, bert_model, mean_vec, cache_path)

    print(f"\n[3] X shape={X.shape}, Y shape={Y.shape}")

    # Train / Val 분리 (90 / 10)
    rng = np.random.default_rng(42)
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

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        optimizer.zero_grad()
        pred = mlp(X_train)
        loss = criterion(pred, Y_train)
        loss.backward()
        optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = criterion(mlp(X_val), Y_val).item()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | train={loss.item():.2f} | val={val_loss:.2f}")

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
    weights_path = DATA_DIR / "mlp_weights.pt"
    torch.save(mlp.state_dict(), weights_path)

    print(f"\n✅ 학습 완료")
    print(f"   저장: {weights_path}")
    print(f"   Best val MSE = {best_val_loss:.2f}  (RGB 0~255 스케일)")
    print(f"\n⚠️  mlp_weights.pt를 팀 카카오톡 또는 구글 드라이브로 공유해주세요.")


if __name__ == "__main__":
    train()
