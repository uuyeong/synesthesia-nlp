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

def find_nearest_word(v_i, norm_mat, candidate_words, v_context=None, alpha=0.0):
    """역변환 벡터에 가장 가까운 후보 단어를 cosine similarity로 찾는다.

    score(word) = α * cos(v_word, v_context) + (1-α) * cos(v_word, v_i)

    Args:
        v_i: 픽셀 역변환 벡터 (768,)
        candidate_matrix: 후보 단어 벡터 행렬 (N, 768)
        candidate_words: 후보 단어 리스트 (N,)
        v_context: 키워드 문맥 벡터 (768,), alpha=0이면 None
        alpha: 키워드 문맥 혼합 비중 (0~1)

    Returns:
        str: 선택된 단어
    """
    # TODO:
    # 1. cosine_similarity(candidate_matrix, v_i) → (N,)
    # 2. alpha > 0이면 cosine_similarity(candidate_matrix, v_context)도 계산
    # 3. score = alpha * sim_context + (1-alpha) * sim_color
    # 4. softmax 샘플링 또는 argmax
    norm_vi = v_i / (np.linalg.norm(v_i) + 1e-8)
    sim_color = norm_mat @ norm_vi

    if alpha > 0 and v_context is not None:
        norm_vc = v_context / np.linalg.norm(v_context)
        sim_context = norm_mat @ norm_vc
    else:
        sim_context = np.zeros_like(sim_color)

    score = alpha * sim_context + (1 - alpha) * sim_color

# argmax 대신 softmax 샘플링
    temperature = 0.5  # 낮을수록 확실한 단어, 높을수록 다양
    score_exp = np.exp((score - score.max()) / temperature)
    prob = score_exp / score_exp.sum()
    idx = np.random.choice(len(candidate_words), p=prob)
    return candidate_words[idx]
    


# ─── 전체 파이프라인 ──────────────────────────────────────────────────────────

def run_reverse(image_path: str, resolution: tuple[int, int] = (8, 8),
                keyword: str | None = None, alpha: float = 0.0) -> list[list[str]]:
    """이미지를 받아 격자 구조의 구조적 시(단어 행렬)를 반환하는 메인 함수.

    Args:
        image_path: 입력 이미지 경로
        resolution: 다운샘플링 해상도 (H, W)
        keyword: 문맥 키워드 (None이면 alpha=0 강제)
        alpha: 키워드 문맥 혼합 비중 (0~1)

    Returns:
        list[list[str]]: H × W 단어 행렬 (격자 구조 유지)
            예) [["burn", "flame", ...], ["ember", "dusk", ...], ...]
    """
    # TODO:
    # 1. load_anchors(), load_candidate_words(), load_candidate_vectors()
    # 2. downsample_image()로 픽셀 배열 획득
    # 3. 각 픽셀에 pixel_to_vector() 적용
    # 4. 키워드가 있으면 get_bert_vector(keyword)로 v_context 계산
    # 5. find_nearest_word()로 단어 선택
    # 6. H × W 행렬 형태로 반환
    tokenizer, model = load_bert()
    mean_vec, A_R, A_G, A_B = load_anchors()
    candidate_words = load_candidate_words()
    candidate_matrix = load_candidate_vectors(candidate_words, tokenizer, model, mean_vec)

    pixels = downsample_image(image_path, resolution)
    H, W = resolution

    v_context = get_bert_vector(keyword, tokenizer, model, mean_vec) if keyword else None
    eff_alpha = alpha if keyword else 0.0

    result = []
    for h in range(H):
        row = []
        for w in range(W):
            v_i = pixel_to_vector(pixels[h, w], A_R, A_G, A_B)
            word = find_nearest_word(v_i, candidate_matrix, candidate_words, v_context, eff_alpha)
            row.append(word)
        result.append(row)
    return result
