"""
app.py — Gradio 인터랙티브 웹 데모
담당: 김민서 (C)

탭 구성:
    탭 1 — 정방향: 텍스트 → 색상 바 + 2D 이미지 + 3D 타워
    탭 2 — 역방향: 이미지 → 구조적 시

실행:
    python src/app.py
"""

# TODO: import gradio as gr
# TODO: from forward_pipeline import run_forward
# TODO: from reverse_pipeline import run_reverse
# TODO: from visualizer import make_color_bar, make_2d_image, make_3d_tower


# ─── 탭 1: 정방향 ─────────────────────────────────────────────────────────────

def forward_tab_handler(text: str, beta: float, gamma: float,
                        ncols: int) -> tuple:
    """Gradio 탭 1 이벤트 핸들러 — 텍스트를 시각화 결과로 변환한다.

    Args:
        text: 사용자 입력 텍스트
        beta: β 슬라이더 값 (0~1, 인간공감각 ←→ AI세계관)
        gamma: γ 슬라이더 값 (0~1, 알파벳 노이즈 강도)
        ncols: 2D 이미지 가로 픽셀 수

    Returns:
        (color_bar_img, pixel_2d_img, plotly_fig)
    """
    # TODO:
    # 1. word_colors = run_forward(text, beta, gamma)
    # 2. color_bar = make_color_bar(word_colors)
    # 3. img_2d = make_2d_image(word_colors, ncols=ncols)
    # 4. fig_3d = make_3d_tower(word_colors)
    # 5. return color_bar, img_2d, fig_3d
    raise NotImplementedError


# ─── 탭 2: 역방향 ─────────────────────────────────────────────────────────────

def reverse_tab_handler(image, resolution_str: str,
                        keyword: str, alpha: float) -> str:
    """Gradio 탭 2 이벤트 핸들러 — 이미지를 구조적 시로 변환한다.

    Args:
        image: Gradio 이미지 입력 (PIL Image 또는 파일 경로)
        resolution_str: 해상도 선택 ("8×8" / "16×16" / "32×32")
        keyword: 키워드 (빈 문자열이면 alpha=0 강제)
        alpha: α 슬라이더 값 (0~1, 색상 ←→ 키워드 문맥)

    Returns:
        str: 생성된 구조적 시 (행 구분은 개행문자)
    """
    # TODO:
    # 1. resolution_str을 (H, W) 튜플로 파싱
    # 2. keyword가 빈 문자열이면 keyword=None, alpha=0
    # 3. poem_grid = run_reverse(image, resolution, keyword, alpha)
    # 4. 행렬을 "단어 단어 단어\n단어 단어 단어\n..." 형태로 join 후 반환
    raise NotImplementedError


# ─── Gradio UI 구성 ───────────────────────────────────────────────────────────

def build_ui():
    """Gradio Blocks UI를 구성하고 반환한다.

    Returns:
        gr.Blocks: 탭 1(정방향) + 탭 2(역방향) 구성의 데모 앱
    """
    # TODO:
    # with gr.Blocks(title="Synesthetic Word Visualizer") as demo:
    #     with gr.Tab("정방향: 텍스트 → 시각화"):
    #         text_input = gr.Textbox(...)
    #         beta_slider = gr.Slider(0, 1, value=0.5, label="β (인간공감각 ←→ AI세계관)")
    #         gamma_slider = gr.Slider(0, 1, value=0.0, label="γ (알파벳 노이즈)")
    #         ncols_slider = gr.Slider(4, 32, value=8, step=1, label="2D 이미지 가로 픽셀")
    #         run_btn = gr.Button("시각화")
    #         color_bar_out = gr.Image(label="색상 바")
    #         img_2d_out = gr.Image(label="2D 이미지")
    #         fig_3d_out = gr.Plot(label="3D 타워")
    #         run_btn.click(forward_tab_handler, ...)
    #     with gr.Tab("역방향: 이미지 → 시"):
    #         image_input = gr.Image(type="filepath", label="이미지 업로드")
    #         resolution_dropdown = gr.Dropdown(["8×8","16×16","32×32"], value="8×8")
    #         keyword_input = gr.Textbox(label="키워드 (선택)", placeholder="없으면 비워두세요")
    #         alpha_slider = gr.Slider(0, 1, value=0.0, label="α (색상 ←→ 키워드)")
    #         reverse_btn = gr.Button("시 생성")
    #         poem_out = gr.Textbox(label="생성된 시", lines=10)
    #         reverse_btn.click(reverse_tab_handler, ...)
    # return demo
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: build_ui().launch()
    raise NotImplementedError
