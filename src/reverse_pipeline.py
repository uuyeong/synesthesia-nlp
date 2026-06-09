"""
reverse_pipeline.py — 역방향 파이프라인 (이미지 → 구조적 시)
담당: 이예원 (B)

흐름:
    이미지
    → 다운샘플링 (PIL, 8×8 / 16×16 / 32×32)
    → 픽셀별 RGB → Anchor 기반 역변환 → v_i (768차원)
    → (선택) α 키워드 문맥 혼합
    → 후보 단어 집합(9,942개)과 cosine similarity → 최근접 단어
    → 격자 구조 유지한 구조적 시 출력
"""

import numpy as np
from pathlib import Path
from forward_pipeline import load_bert, load_anchors, get_bert_vector
from PIL import Image
DATA_DIR = Path(__file__).parent.parent / "data"


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_candidate_words() -> list[str]:
    """역방향 후보 단어 집합(poetry_candidate_words.txt)을 로드한다.

    Returns:
        list[str]: 9,942개 시적 단어 리스트
    """
    path = DATA_DIR / "poetry_candidate_words.txt"
    return path.read_text(encoding="utf-8").splitlines()


def load_candidate_vectors(candidate_words: list[str], tokenizer, model,
                           mean_vec: np.ndarray) -> np.ndarray:
    """후보 단어 전체의 BERT Mean Centering 벡터를 계산하거나 캐시에서 로드한다.

    Args:
        candidate_words: 후보 단어 리스트
        tokenizer, model: BERT
        mean_vec: Mean Centering 평균 벡터 (768,)

    Returns:
        np.ndarray: shape (N, 768) — 후보 단어별 벡터 행렬
    """
    cache = DATA_DIR / "candidate_vectors.npy"
    if cache.exists():
        return np.load(cache)
    vecs = np.array([get_bert_vector(w, tokenizer, model, mean_vec) for w in candidate_words])
    np.save(cache, vecs)
    return vecs
    


# ─── 이미지 처리 ──────────────────────────────────────────────────────────────

def downsample_image(image_path: str, resolution: tuple[int, int]) -> np.ndarray:
    """이미지를 지정 해상도로 다운샘플링하여 RGB 픽셀 배열을 반환한다.

    Args:
        image_path: 입력 이미지 경로 (JPG/PNG)
        resolution: (height, width) 예) (8, 8), (16, 16), (32, 32)

    Returns:
        np.ndarray: shape (H, W, 3), dtype uint8, 범위 0~255
    """
    # TODO: PIL Image.open().resize(resolution, LANCZOS).convert("RGB")
    H, W = resolution
    img = Image.open(image_path).resize((W, H), Image.LANCZOS).convert("RGB")
    return np.array(img)


# ─── 역변환 ──────────────────────────────────────────────────────────────────

def pixel_to_vector(rgb: np.ndarray, A_R: np.ndarray,
                    A_G: np.ndarray, A_B: np.ndarray) -> np.ndarray:
    """픽셀 RGB값을 Anchor 기반 역변환으로 BERT 벡터 공간에 매핑한다.

    v_i = (R/255)*A_R + (G/255)*A_G + (B/255)*A_B

    Args:
        rgb: RGB 값 (3,), 범위 0~255
        A_R, A_G, A_B: RGB 앵커 벡터 (768,) 각각

    Returns:
        np.ndarray: 역변환된 벡터 (768,)
    """
    # TODO: 수식 그대로 구현
    R, G, B = rgb[0]/255, rgb[1]/255, rgb[2]/255
    return R*A_R + G*A_G + B*A_B


# ─── 단어 검색 ────────────────────────────────────────────────────────────────

def find_nearest_word(v_i, norm_mat, candidate_words, v_context=None, alpha=0.0,
                      temperature=0.5):
    """역변환 벡터에 가장 가까운 후보 단어를 cosine similarity로 찾는다.

    score(word) = α * cos(v_word, v_context) + (1-α) * cos(v_word, v_i)

    Args:
        v_i: 픽셀 역변환 벡터 (768,)
        norm_mat: 행 단위 L2 정규화된 후보 단어 행렬 (N, 768).
                  호출 측에서 candidate_matrix를 미리 정규화해 전달해야 한다.
        candidate_words: 후보 단어 리스트 (N,)
        v_context: 키워드 문맥 벡터 (768,), alpha=0이면 None
        alpha: 키워드 문맥 혼합 비중 (0~1)
        temperature: softmax 온도. 낮을수록 확실한 단어(일관성↑),
                     높을수록 다양(무작위↑). 일관성 옵션에서 낮춰서 전달한다.

    Returns:
        str: 선택된 단어
    """
    norm_vi = v_i / (np.linalg.norm(v_i) + 1e-8)
    sim_color = norm_mat @ norm_vi

    if alpha > 0 and v_context is not None:
        norm_vc = v_context / (np.linalg.norm(v_context) + 1e-8)
        sim_context = norm_mat @ norm_vc
    else:
        sim_context = np.zeros_like(sim_color)

    score = alpha * sim_context + (1 - alpha) * sim_color

    # 상위 top_k 후보로 좁힌 뒤 softmax 샘플링 (전체 9,942개 softmax는 사실상
    # 균등 랜덤이 되는 문제 때문). _run_reverse_core 도 동일 방식을 쓴다.
    return candidate_words[_sample_idx(score, temperature, top_k=40)]


# ─── 일관성(coherence) — 모델 벡터 공간 기반 후처리 ──────────────────────────

def _box_blur_grid(grid: np.ndarray) -> np.ndarray:
    """(H, W, C) 벡터 격자에 3×3 평균 블러를 적용한다 (가장자리는 복제)."""
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(grid)
    H, W = grid.shape[:2]
    for di in range(3):
        for dj in range(3):
            acc += padded[di:di + H, dj:dj + W, :]
    return acc / 9.0


def apply_coherence(vec_grid: np.ndarray, coherence: float) -> np.ndarray:
    """픽셀 벡터 격자에 공간 스무딩을 적용한다(국소 색 흐름 안정화).

    인접 픽셀 벡터를 섞어 이웃끼리 색이 급변하지 않게 한다. 의미적 결속은
    `_run_reverse_core`의 이웃 단어 전파가 담당하고, 여기서는 색 흐름만 다듬는다.

    Args:
        vec_grid: (H, W, 768) 픽셀 역변환 벡터 격자
        coherence: 0(기존 날것 유지) ~ 1(최대 스무딩)

    Returns:
        (H, W, 768) 스무딩된 벡터 격자
    """
    if coherence <= 0:
        return vec_grid
    # 과한 붕괴를 막기 위해 스무딩 강도를 0.6까지만 반영
    s = 0.6 * coherence
    return (1 - s) * vec_grid + s * _box_blur_grid(vec_grid)


def _sample_idx(score: np.ndarray, temperature: float, top_k: int = 40) -> int:
    """점수 상위 top_k 후보 중에서만 softmax 온도 샘플링으로 인덱스를 고른다.

    후보가 9,942개라 전체에 softmax를 걸면(과거 방식) 점수 차가 온도에 비해
    작아 사실상 균등 랜덤이 됐다 — 좋은 색·테마 후보가 있어도 무관한 단어가
    뽑히는 주된 원인. 상위 top_k로 좁힌 뒤 낮은 온도로 샘플링해 색 충실도와
    다양성을 모두 확보한다.
    """
    t = max(temperature, 1e-3)
    n = len(score)
    if 0 < top_k < n:
        cand = np.argpartition(score, -top_k)[-top_k:]
    else:
        cand = np.arange(n)
    s = score[cand]
    exp = np.exp((s - s.max()) / t)
    prob = exp / exp.sum()
    return int(np.random.choice(cand, p=prob))



# ─── 전체 파이프라인 ──────────────────────────────────────────────────────────

def _run_reverse_core(image_path: str, resolution: tuple[int, int],
                      keyword: str | None, alpha: float,
                      coherence: float = 0.0) -> dict:
    """Run reverse generation and return the full internal result payload."""
    tokenizer, model = load_bert()
    mean_vec, A_R, A_G, A_B = load_anchors()
    candidate_words = load_candidate_words()
    candidate_matrix = load_candidate_vectors(candidate_words, tokenizer, model, mean_vec)

    # cosine similarity를 위해 후보 행렬을 1회만 행 단위 L2 정규화한다.
    # (이전엔 raw 행렬이 그대로 전달되어 cosine이 아닌 단순 내적이 계산됐음)
    norm_mat = candidate_matrix / (
        np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-8
    )

    pixels = downsample_image(image_path, resolution)
    H, W = resolution

    v_context = get_bert_vector(keyword, tokenizer, model, mean_vec) if keyword else None
    eff_alpha = alpha if keyword else 0.0

    # 픽셀 역변환 벡터 격자를 만든 뒤 색 흐름 스무딩(coherence)을 적용한다.
    # coherence=0이면 격자가 그대로라 기존 '날것' 동작과 완전히 동일하다.
    coherence = float(np.clip(coherence, 0.0, 1.0))
    vec_grid = np.empty((H, W, mean_vec.shape[0]), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            vec_grid[h, w] = pixel_to_vector(pixels[h, w], A_R, A_G, A_B)
    vec_grid = apply_coherence(vec_grid, coherence)

    # 상위 top_k 후보로 좁혀 샘플링한다. 일관성이 높을수록 후보를 좁히고(40→10)
    # 온도를 낮춰(0.22→0.10) 더 확실·결속된 단어를, 낮을수록 색 충실 + 다양성.
    temperature = 0.22 - 0.12 * coherence
    top_k = max(8, int(round(40 - 30 * coherence)))

    # 키워드 문맥 유사도(선택)는 1회만 계산.
    kw_sim = None
    if eff_alpha > 0 and v_context is not None:
        norm_vc = v_context / (np.linalg.norm(v_context) + 1e-8)
        kw_sim = norm_mat @ norm_vc

    result = []
    mapping_rows = []
    chosen_idx = np.full((H, W), -1, dtype=int)  # 이웃 단어 전파용 선택 인덱스
    for h in range(H):
        row = []
        for w in range(W):
            pixel_rgb = pixels[h, w]
            v_i = vec_grid[h, w]
            color_sim = norm_mat @ (v_i / (np.linalg.norm(v_i) + 1e-8))
            score = color_sim

            # 이웃 단어 의미 전파 — 이미 놓인 좌·상단 단어 벡터에 의미적으로
            # 가깝도록 유도해 국소 테마가 격자를 따라 번지게 한다(LLM 없이 결속).
            if coherence > 0:
                neigh = [idx for idx in (chosen_idx[h, w - 1] if w > 0 else -1,
                                         chosen_idx[h - 1, w] if h > 0 else -1)
                         if idx >= 0]
                if neigh:
                    ctx = norm_mat[neigh].mean(axis=0)
                    ctx /= (np.linalg.norm(ctx) + 1e-8)
                    # 이웃 가중치를 0.6까지만 — 색이 항상 ≥0.4 영향을 유지해
                    # 모든 픽셀이 한 단어로 붕괴하지 않도록 한다.
                    nbw = 0.6 * coherence
                    score = (1 - nbw) * color_sim + nbw * (norm_mat @ ctx)

            # 키워드 문맥은 그 위에 추가로 혼합.
            if kw_sim is not None:
                score = (1 - eff_alpha) * score + eff_alpha * kw_sim

            idx = _sample_idx(score, temperature, top_k)
            chosen_idx[h, w] = idx
            word = candidate_words[idx]
            row.append(word)
            mapping_rows.append({
                "row": h + 1,
                "col": w + 1,
                "rgb": [int(channel) for channel in pixel_rgb],
                "hex": "#{:02X}{:02X}{:02X}".format(*pixel_rgb),
                "word": word,
            })
        result.append(row)

    return {
        "word_grid": result,
        "pixels": pixels,
        "mapping_rows": mapping_rows,
    }


def run_reverse(image_path: str, resolution: tuple[int, int] = (8, 8),
                keyword: str | None = None, alpha: float = 0.0,
                coherence: float = 0.0) -> list[list[str]]:
    """이미지를 받아 격자 구조의 구조적 시(단어 행렬)를 반환하는 메인 함수.

    Args:
        image_path: 입력 이미지 경로
        resolution: 다운샘플링 해상도 (H, W)
        keyword: 문맥 키워드 (None이면 alpha=0 강제)
        alpha: 키워드 문맥 혼합 비중 (0~1)
        coherence: 일관성 강도 (0=날것, 1=최대 결속). 모델 벡터 공간 후처리.

    Returns:
        list[list[str]]: H × W 단어 행렬 (격자 구조 유지)
            예) [["burn", "flame", ...], ["ember", "dusk", ...], ...]
    """
    return _run_reverse_core(image_path, resolution, keyword, alpha, coherence)["word_grid"]


def run_reverse_with_details(image_path: str, resolution: tuple[int, int] = (8, 8),
                             keyword: str | None = None, alpha: float = 0.0,
                             coherence: float = 0.0) -> dict:
    """이미지를 받아 단어 행렬, 다운샘플 픽셀, 색상-단어 매핑을 반환한다.

    UI에서 단순화된 이미지와 색상-단어 대응표를 표시할 때 사용한다.
    """
    return _run_reverse_core(image_path, resolution, keyword, alpha, coherence)
