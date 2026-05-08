"""
visualizer.py — 2D 픽셀 이미지 및 3D 색상 타워 시각화
담당: 김민서 (C)

기능:
    - 2D 이미지: 단어별 RGB를 raster scan으로 배열한 픽셀 이미지
    - 3D 타워: Plotly scatter3d, XY=RGB 삼각형 좌표, Z=단어 순서
    - 색상 바: 단어별 색상을 수평으로 나열한 HTML/이미지 스트립
"""

import numpy as np


# ─── 2D 이미지 ────────────────────────────────────────────────────────────────

def make_2d_image(word_colors: list[dict], ncols: int = 8) -> "PIL.Image.Image":
    """단어별 RGB를 raster scan으로 배열하여 2D 픽셀 이미지를 생성한다.

    한 단어 = 한 픽셀. 단어 수가 ncols의 배수가 아닌 경우 (128,128,128)으로 패딩.

    Args:
        word_colors: forward_pipeline.run_forward() 반환값
            [{"word": str, "rgb_out": [R,G,B]}, ...]
        ncols: 이미지 가로 픽셀 수 (= 한 행의 단어 수)

    Returns:
        PIL.Image.Image: RGB 모드 이미지 (픽셀 단위)
    """
    # TODO:
    # 1. rgb_out 값을 (H, W, 3) numpy 배열로 reshape
    # 2. PIL.Image.fromarray(arr.astype(np.uint8))로 변환
    raise NotImplementedError


def make_color_bar(word_colors: list[dict], pixel_width: int = 40,
                   pixel_height: int = 40) -> "PIL.Image.Image":
    """단어별 색상을 수평으로 나열한 색상 바 이미지를 생성한다.

    Args:
        word_colors: forward_pipeline.run_forward() 반환값
        pixel_width: 단어당 색상 블록의 가로 픽셀 수
        pixel_height: 색상 바의 세로 픽셀 수

    Returns:
        PIL.Image.Image: 가로 = N*pixel_width, 세로 = pixel_height
    """
    # TODO: 각 단어별 색상 블록을 numpy 배열로 이어 붙인 뒤 PIL 변환
    raise NotImplementedError


# ─── 3D 타워 ─────────────────────────────────────────────────────────────────

def rgb_to_barycentric(r: float, g: float, b: float) -> tuple[float, float]:
    """RGB 값을 등변삼각형 무게중심 좌표(x, y)로 변환한다.

    삼각형 꼭짓점: R=(0,0), G=(1,0), B=(0.5, √3/2)

    Args:
        r, g, b: 0~255 범위 RGB 값

    Returns:
        (x, y): 삼각형 내부 좌표
    """
    # TODO:
    # total = r + g + b (0이면 (1/3, √3/6) 반환)
    # x = g/total + b/total * 0.5
    # y = b/total * (3**0.5 / 2)
    raise NotImplementedError


def make_3d_tower(word_colors: list[dict]) -> "plotly.graph_objects.Figure":
    """단어별 RGB를 3D 색상 타워로 시각화하는 Plotly figure를 반환한다.

    XY: RGB 삼각형 무게중심 좌표, Z: 단어 순서 (시간축)
    Hover: 단어명 + RGB_syn / RGB_uni / RGB_out 표시

    Args:
        word_colors: forward_pipeline.run_forward() 반환값
            [{"word": str, "rgb_syn": [...], "rgb_uni": [...], "rgb_out": [...]}, ...]

    Returns:
        plotly.graph_objects.Figure: 인터랙티브 3D scatter plot
    """
    # TODO:
    # 1. 각 단어별 rgb_to_barycentric()으로 XY 계산
    # 2. Z = 단어 인덱스 (0, 1, 2, ...)
    # 3. marker color = rgb_out 값
    # 4. hovertext = f"{word}<br>syn={rgb_syn}<br>uni={rgb_uni}<br>out={rgb_out}"
    # 5. go.Scatter3d로 figure 생성
    raise NotImplementedError
