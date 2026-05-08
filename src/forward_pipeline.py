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

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── 모델 로드 ────────────────────────────────────────────────────────────────

def load_bert():
    """BERT 토크나이저와 모델을 로드한다.

    Returns:
        tokenizer: BertTokenizer
        model: BertModel (eval 모드)
    """
    # TODO: transformers에서 BertTokenizer, BertModel 로드
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased").eval()
    raise NotImplementedError


def load_anchors():
    """저장된 BERT Mean Centering 벡터와 RGB 앵커 벡터를 로드한다.

    Returns:
        mean_vec (np.ndarray): shape (768,)
        A_R (np.ndarray): shape (768,)
        A_G (np.ndarray): shape (768,)
        A_B (np.ndarray): shape (768,)
    """
    # TODO: DATA_DIR에서 .npy 파일 4개 로드
    raise NotImplementedError


def load_mlp(weights_path: str):
    """학습된 MLP 가중치를 로드한다.

    Args:
        weights_path: mlp_weights.pt 경로

    Returns:
        model: MLP (eval 모드)
    """
    # TODO: MLP 클래스 정의 후 torch.load로 로드
    raise NotImplementedError


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
    # TODO:
    # 1. tokenizer(word, return_tensors='pt')
    # 2. model(**inputs).last_hidden_state[0, 1:-1, :]  → 서브워드 토큰들
    # 3. 서브워드가 여러 개이면 평균 풀링
    # 4. - mean_vec 적용
    raise NotImplementedError


def tokenize_text(text: str, tokenizer) -> list[str]:
    """텍스트를 단어 단위로 분리한다 (단어 경계 기준, 구두점 분리 포함).

    Args:
        text: 입력 텍스트

    Returns:
        list[str]: 단어 리스트
    """
    # TODO: 단순 split 또는 nltk word_tokenize 사용
    raise NotImplementedError


# ─── 색상 계산 ────────────────────────────────────────────────────────────────

def rgb_syn(v_i: np.ndarray, mlp) -> np.ndarray:
    """방법 1: MLP (768→32→3)로 RGB_syn을 계산한다.

    Args:
        v_i: Mean Centering 적용된 BERT 벡터 (768,)
        mlp: 학습된 MLP 모델

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255
    """
    # TODO: torch.no_grad()로 MLP forward pass
    raise NotImplementedError


def rgb_uni(v_i: np.ndarray, A_R: np.ndarray, A_G: np.ndarray, A_B: np.ndarray) -> np.ndarray:
    """방법 2: Cosine Similarity 기반 RGB_uni를 계산한다.

    cos(v_i, A_R), cos(v_i, A_G), cos(v_i, A_B)를 계산하고
    0~255 범위로 정규화한다.

    Args:
        v_i: Mean Centering 적용된 BERT 벡터 (768,)
        A_R, A_G, A_B: RGB 앵커 벡터 (768,) 각각

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255
    """
    # TODO:
    # 1. cos_r, cos_g, cos_b = cosine_similarity(v_i, A_R/G/B)
    # 2. shift to [0, 1] then scale to [0, 255]
    raise NotImplementedError


def get_grapheme_color(word: str, eagleman: dict) -> np.ndarray:
    """단어의 첫 글자를 Eagleman 데이터에서 조회하여 RGB_grapheme을 반환한다.

    근거: 공감각 연구에서 단어 색상은 첫 글자 색상이 지배적
    (Rich et al., 2005; Simner et al., 2006)

    Args:
        word: 입력 단어
        eagleman: {알파벳 → RGB np.ndarray} 딕셔너리

    Returns:
        np.ndarray: RGB 값 (3,), 범위 0~255. 알파벳 이외 문자는 (128,128,128) 반환.
    """
    # TODO: word[0].upper()로 첫 글자 추출 후 eagleman dict 조회
    raise NotImplementedError


def blend(rgb_syn: np.ndarray, rgb_uni: np.ndarray,
          rgb_grapheme: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """β 혼합과 γ 알파벳 노이즈를 적용하여 최종 RGB_out을 계산한다.

    RGB_final = β * RGB_syn + (1-β) * RGB_uni
    RGB_out   = RGB_final + γ * (RGB_grapheme - RGB_final)

    Args:
        rgb_syn: MLP 출력 (3,)
        rgb_uni: Cosine Similarity 출력 (3,)
        rgb_grapheme: Eagleman 첫 글자 색상 (3,)
        beta: 공감각 비중 (0~1), 1이면 MLP만 사용
        gamma: 알파벳 노이즈 강도 (0~1)

    Returns:
        np.ndarray: 최종 RGB 값 (3,), 범위 0~255
    """
    # TODO: 수식 그대로 구현, np.clip(result, 0, 255)로 마무리
    raise NotImplementedError


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
    # TODO:
    # 1. load_bert(), load_anchors(), load_mlp() 호출
    # 2. tokenize_text()로 단어 분리
    # 3. 각 단어에 대해 get_bert_vector() → rgb_syn() + rgb_uni() → blend()
    # 4. 결과 dict 리스트 반환
    raise NotImplementedError
