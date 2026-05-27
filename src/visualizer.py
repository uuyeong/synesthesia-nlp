"""
visualizer.py — 2D 픽셀 이미지 및 3D 색상 타워 시각화
담당: 김민서 (C)

기능:
    - 2D 이미지: 단어별 RGB를 raster scan으로 배열한 픽셀 이미지
    - 3D 타워: Plotly scatter3d, XY=RGB 삼각형 좌표, Z=단어 순서
    - 색상 바: 단어별 색상을 수평으로 나열한 HTML/이미지 스트립
"""

import numpy as np
from PIL import Image
import plotly.graph_objects as go



# ─── 2D 이미지 ────────────────────────────────────────────────────────────────
'''
def make_2d_image(word_colors: list[dict], ncols: int = 10) -> Image.Image:
    """단어별 RGB를 배열하여 2D 이미지를 생성하고, 가로 1024px 크기의 정사각형 블록으로 확대한다."""

    target_width = 1024

    # 예외 처리: 데이터가 없는 경우 회색 빈 이미지 반환
    if not word_colors:
        square_size = target_width // ncols
        return Image.new("RGB", (target_width, square_size), (128, 128, 128))

    # 1. rgb_out 값 추출 (키가 없으면 회색으로 처리)
    rgbs = [item.get("rgb_out", [128, 128, 128]) for item in word_colors]

    # 2. 패딩(Padding) 처리: 단어 수가 ncols의 배수가 되도록 회색(128,128,128) 추가
    remainder = len(rgbs) % ncols
    if remainder != 0:
        padding_size = ncols - remainder
        rgbs.extend([[128, 128, 128]] * padding_size)

    # 3. (H, W, 3) numpy 배열로 reshape (1단어 = 1픽셀 크기의 원본)
    arr = np.array(rgbs, dtype=np.uint8)
    arr = arr.reshape((-1, ncols, 3))
    base_img = Image.fromarray(arr)

    # 4. 가로 1024 픽셀 고정 및 정사각형 비율에 맞춘 세로 길이 계산
    nrows = arr.shape[0]
    target_height = int(target_width * (nrows / ncols))

    # 5. 블러(뭉개짐) 없이 픽셀 아트를 그대로 확대하기 위해 NEAREST 사용
    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST # 구버전 PIL 호환용

    final_img = base_img.resize((target_width, target_height), resample=resample_method)

    return final_img
'''
import math
import numpy as np
from PIL import Image

def make_2d_image(word_colors: list[dict], unit: str = "character") -> Image.Image:
    """Render either one character or one word as each source pixel."""

    target_size = 1024 # 가로세로 무조건 1024 고정

    # 예외 처리: 데이터가 없는 경우 빈 회색 이미지 반환
    if not word_colors:
        return Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # 1. 보기 모드에 따라 단어 또는 글자 RGB를 추출한다.
    if unit == "word":
        rgbs = [item.get("rgb_out", [0, 0, 0]) for item in word_colors]
    else:
        rgbs = []
        for item in word_colors:
            char_rgbs = item.get("char_rgbs")
            if char_rgbs:
                rgbs.extend(char_rgbs)
            else:
                word = item.get("word", "")
                rgb = item.get("rgb_out", [0, 0, 0])
                rgbs.extend([rgb] * max(1, len(word)))

    if not rgbs:
        return Image.new("RGB", (target_size, target_size), (0, 0, 0))

    L = len(rgbs)

    # 2. 완벽한 정사각형 그리드를 위한 픽셀 수(N) 계산
    # 데이터 개수(L)의 제곱근을 올림하여 N x N 배열을 만듭니다.
    # 예: 100글자 -> 10x10 / 120글자 -> 11x11
    N = math.ceil(math.sqrt(L))
    N = max(1, N) # 최소 1x1 보장

    # 3. 패딩(Padding) 처리: N x N 개수에 모자란 만큼 빈칸을 회색(128,128,128)으로 채움
    padding_size = (N * N) - L
    if padding_size > 0:
        rgbs.extend([[0, 0, 0]] * padding_size)

    # 4. (N, N, 3) 형태의 numpy 3차원 배열로 완벽한 1:1 reshape
    arr = np.array(rgbs, dtype=np.uint8)
    arr = arr.reshape((N, N, 3))
    base_img = Image.fromarray(arr)

    # 5. 블러(뭉개짐) 없이 1024x1024 픽셀 아트로 꽉 차게 확대
    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST # 구버전 PIL 호환용

    final_img = base_img.resize((target_size, target_size), resample=resample_method)

    return final_img

def make_color_bar(word_colors: list[dict]) -> Image.Image:
    """단어별 RGB를 가로로 나열하여, UI 칸에 꽉 차는 두꺼운 Color Bar를 생성한다."""

    target_width = 1024
    target_height = 120  # 세로 길이를 충분히 크게 고정 (Gradio height=500에 대응)

    # 예외 처리: 데이터가 없는 경우 회색 이미지 반환
    if not word_colors:
        return Image.new("RGB", (target_width, target_height), (0, 0, 0))

    # 1. 모든 단어(또는 글자)의 RGB 값 추출
    rgbs = [item.get("rgb_out", [0, 0, 0]) for item in word_colors]

    # 2. 1줄짜리 원본 numpy 배열 생성 (Shape: 1 x 단어수 x 3)
    arr = np.array([rgbs], dtype=np.uint8)
    base_img = Image.fromarray(arr)

    # 3. 가로 1024px, 세로 512px로 강제 확대
    # NEAREST를 사용하여 색상 경계를 뚜렷하게 유지합니다.
    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST

    final_bar = base_img.resize((target_width, target_height), resample=resample_method)

    return final_bar


# ─── 3D 타워 ─────────────────────────────────────────────────────────────────

def rgb_to_barycentric(r: float, g: float, b: float) -> tuple[float, float]:
    """RGB 값을 등변삼각형 무게중심 좌표(x, y)로 변환한다."""

    total = r + g + b

    # RGB가 모두 0인 경우 (검은색), 0으로 나누는 에러 방지를 위해 삼각형의 정중앙 반환
    if total == 0:
        return (1/3, (3**0.5) / 6)

    x = (g / total) + (b / total) * 0.5
    y = (b / total) * ((3**0.5) / 2)

    return (x, y)


def make_3d_tower(word_colors: list[dict]) -> go.Figure:
    """
    단어별 RGB를 3D 색상 타워로 시각화
    Plotly Slider 기반 인터랙션 버전
    """

    if not word_colors:
        return go.Figure()

    # ─────────────────────────────────────────────
    # 데이터 준비
    # ─────────────────────────────────────────────

    xs, ys, zs = [], [], []
    marker_colors = []
    hover_texts = []

    total_words = len(word_colors)

    for z, item in enumerate(word_colors):
        r, g, b = item.get("rgb_out", [128, 128, 128])

        x, y = rgb_to_barycentric(r, g, b)
        xs.append(x)
        ys.append(y)
        zs.append(z)

        word = item.get("word", "")
        rgb_syn = item.get("rgb_syn", [])
        rgb_uni = item.get("rgb_uni", [])
        rgb_out = [int(r), int(g), int(b)]

        hover_text = f"<b>{word}</b><br>Syn: {rgb_syn}<br>Uni: {rgb_uni}<br>Out: {rgb_out}"
        hover_texts.append(hover_text)

        marker_colors.append(f"rgb({rgb_out[0]}, {rgb_out[1]}, {rgb_out[2]})")

    # ─────────────────────────────────────────────
    # 초기 Figure
    # ─────────────────────────────────────────────

    fig = go.Figure()

    fig.add_trace(

        go.Scatter3d(

            x=[xs[0]],
            y=[ys[0]],
            z=[zs[0]],

            mode="markers+lines",

            marker=dict(size=3, color=marker_colors[:1], opacity=0.9, line=dict(width=1, color="#666666")),
            line=dict(color='rgba(200, 200, 200, 0.2)', width=0.5),
            text=hover_texts[:1],
            hoverinfo='text'
        )
    )

    # ─────────────────────────────────────────────
    # Slider Steps 생성
    # ─────────────────────────────────────────────

    max_steps = 100
    step_size = max(1, total_words // max_steps)

    slider_steps = []

    for i in range(1, total_words + 1, step_size):

        step = dict(

            method="update",

            args=[

                {
                    "x": [xs[:i]],
                    "y": [ys[:i]],
                    "z": [zs[:i]],

                    "text": [hover_texts[:i]],

                    "marker.color": [marker_colors[:i]],
                }
            ],

            label=str(i)
        )

        slider_steps.append(step)

    # 마지막 강제 추가
    if slider_steps[-1]["label"] != str(total_words):

        slider_steps.append(

            dict(
                method="update",

                args=[

                    {
                        "x": [xs],
                        "y": [ys],
                        "z": [zs],

                        "text": [hover_texts],

                        "marker.color": [marker_colors],
                    }
                ],

                label=str(total_words)
            )
        )

    # ─────────────────────────────────────────────
    # 레이아웃
    # ─────────────────────────────────────────────

    fig.update_layout(

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

        paper_bgcolor="#101010",
        plot_bgcolor="#101010",
        font=dict(color="#d0d0d0"),

        scene=dict(
            bgcolor="#101010",

            xaxis=dict(
                title="X",
                backgroundcolor="#1a1a1a",
                gridcolor="#404040",
                linecolor="#606060",
                zerolinecolor="#606060",
            ),

            yaxis=dict(
                title="Y",
                backgroundcolor="#1a1a1a",
                gridcolor="#404040",
                linecolor="#606060",
                zerolinecolor="#606060",
            ),

            zaxis=dict(
                title="Time",
                range=[0, total_words],
                backgroundcolor="#1a1a1a",
                gridcolor="#404040",
                linecolor="#606060",
                zerolinecolor="#606060",
            ),
        ),

        sliders=[

            dict(

                active=0,

                currentvalue=dict(
                    prefix="Words: ",
                    font=dict(color="#d0d0d0"),
                ),

                pad=dict(
                    t=30
                ),

                steps=slider_steps,
                bgcolor="#242424",
                bordercolor="#505050",
                font=dict(color="#d0d0d0"),
            )
        ]
    )

    return fig
