"""
app.py — Gradio 인터랙티브 웹 데모
담당: 김민서 (C)

탭 구성:
    탭 1 — 정방향: 텍스트 → 색상 바 + 2D 이미지 + 3D 타워
    탭 2 — 역방향: 이미지 → 구조적 시
    탭 3 — 순환 실험: 텍스트 → 정방향 → 색상 이미지 → 역방향 → 새로운 시

실행:
    python src/app.py
"""

import html
import math
import os
import tempfile
import gradio as gr
from PIL import Image
from forward_pipeline import reblend_forward_results, run_forward
from reverse_pipeline import run_reverse_with_details
from visualizer import make_color_bar, make_2d_image, make_3d_tower


APP_THEME = gr.themes.Monochrome()

APP_CSS = """
:root {
    --surface-0: #191919;
    --surface-1: #191919;
    --surface-2: #252525;
    --surface-3: #333333;
    --field: #0d0d0d;
    --border: #4a4a4a;
    --border-strong: #747474;
    --text: #f4f4f4;
    --text-muted: #c6c6c6;
    --control-hover: #3f3f3f;
    --control-active: #f1f1f1;
    --control-active-text: #111111;
}

html, body, .gradio-container {
    background: var(--surface-0) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text);
    font-family: "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕",
                 "Noto Sans KR", "Segoe UI", Arial, sans-serif !important;
}

.gradio-container {
    min-height: 100vh;
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 24px 32px !important;
    box-sizing: border-box !important;
}

.gradio-container * {
    font-family: inherit !important;
}

.gradio-container .prose,
.gradio-container label,
.gradio-container span,
.gradio-container p,
.gradio-container h1 {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text);
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container fieldset {
    background: var(--surface-1) !important;
    border-color: var(--border) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: var(--field) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
}

.gradio-container button,
.gradio-container [role="tab"],
.gradio-container [role="radio"] {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    transition: background-color 160ms ease, border-color 160ms ease,
                color 160ms ease, transform 160ms ease, box-shadow 160ms ease !important;
}

.gradio-container button:hover,
.gradio-container [role="tab"]:hover {
    background: var(--control-hover) !important;
    border-color: var(--border-strong) !important;
}

.gradio-container button.primary {
    background: #303030 !important;
    border-color: #686868 !important;
}

.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="radio"][aria-checked="true"] {
    background: #2a2a2a !important;
    border-color: #777777 !important;
}

.gradio-container input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 20px !important;
    min-height: 20px !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    accent-color: #d8d8d8;
}

.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    height: 4px;
    border-radius: 999px;
    background: #4a4a4a;
}

.gradio-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    margin-top: -6px;
    border: 2px solid #101010;
    border-radius: 50%;
    background: #e5e5e5;
    box-shadow: 0 0 0 1px #6b6b6b;
}

.gradio-container input[type="range"]::-moz-range-track {
    height: 4px;
    border-radius: 999px;
    background: #4a4a4a;
}

.gradio-container input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border: 2px solid #101010;
    border-radius: 50%;
    background: #e5e5e5;
    box-shadow: 0 0 0 1px #6b6b6b;
}

#generate-btn,
#generate-btn button {
    background: #303030 !important;
    border-color: #686868 !important;
    transition: background-color 160ms ease, border-color 160ms ease,
                transform 160ms ease, box-shadow 160ms ease !important;
}

#generate-btn:hover,
#generate-btn button:hover {
    background: #484848 !important;
    border-color: #9a9a9a !important;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.38) !important;
    transform: translateY(-1px);
}

#generate-btn:active,
#generate-btn button:active {
    background: #202020 !important;
    box-shadow: none !important;
    transform: translateY(1px);
}

#reverse-generate-btn,
#reverse-generate-btn button {
    background: #303030 !important;
    border-color: #686868 !important;
    cursor: pointer !important;
    transition: background-color 160ms ease, border-color 160ms ease,
                transform 160ms ease, box-shadow 160ms ease !important;
}

#reverse-generate-btn:hover,
#reverse-generate-btn button:hover {
    background: #484848 !important;
    border-color: #9a9a9a !important;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.38) !important;
    transform: translateY(-1px);
}

#reverse-generate-btn:active,
#reverse-generate-btn button:active {
    background: #202020 !important;
    box-shadow: none !important;
    transform: translateY(1px);
}

#resolution-dropdown,
#resolution-dropdown *,
#resolution-dropdown input,
#resolution-dropdown button,
#resolution-dropdown [role="button"],
#resolution-dropdown [role="combobox"] {
    cursor: pointer !important;
}

#image-unit-toggle label,
#image-unit-toggle [role="radio"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 118px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface-2) !important;
    text-align: center !important;
    transition: background-color 160ms ease, border-color 160ms ease,
                color 160ms ease, transform 160ms ease !important;
}

#image-unit-toggle label span,
#image-unit-toggle [role="radio"] span {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
}

#image-unit-toggle input[type="radio"],
#image-unit-toggle label input[type="radio"],
#image-unit-toggle [role="radio"]::before,
#image-unit-toggle [role="radio"]::after {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    appearance: none !important;
    -webkit-appearance: none !important;
}

#image-unit-toggle label:hover,
#image-unit-toggle [role="radio"]:hover {
    background: var(--control-hover) !important;
    border-color: #858585 !important;
    transform: translateY(-1px);
}

#image-unit-toggle label:has(input:checked),
#image-unit-toggle label.selected,
#image-unit-toggle [role="radio"][aria-checked="true"] {
    background: var(--control-active) !important;
    border-color: var(--control-active) !important;
}

#image-unit-toggle label:has(input:checked) *,
#image-unit-toggle label.selected *,
#image-unit-toggle [role="radio"][aria-checked="true"] * {
    color: var(--control-active-text) !important;
    -webkit-text-fill-color: var(--control-active-text) !important;
}

.reverse-map-wrap {
    max-height: 440px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface-1);
}

.reverse-map-table {
    width: 100%;
    border-collapse: collapse;
    color: var(--text);
    font-size: 13px;
}

.reverse-map-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    padding: 10px 12px;
    background: #222222;
    border-bottom: 1px solid var(--border-strong);
    color: var(--text);
    text-align: left;
}

.reverse-map-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #292929;
    color: var(--text);
}

.reverse-map-table tr:hover td {
    background: #252525;
}

.reverse-swatch {
    display: inline-block;
    width: 22px;
    height: 22px;
    border: 1px solid #7a7a7a;
    border-radius: 4px;
    vertical-align: middle;
}

#reverse-results-column {
    gap: 18px !important;
}

#reverse-results-column textarea {
    min-height: 260px !important;
}

#reverse-results-column img {
    image-rendering: pixelated;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 16px 14px !important;
    }

    #reverse-layout {
        flex-direction: column !important;
        gap: 18px !important;
    }

    #reverse-layout > div {
        min-width: 100% !important;
        width: 100% !important;
    }

    #image-unit-toggle label,
    #image-unit-toggle [role="radio"] {
        width: 100% !important;
        justify-content: center !important;
    }
}
"""


# ─── 탭 1: 정방향 ─────────────────────────────────────────────────────────────

def render_forward_outputs(word_colors: list[dict], image_unit: str) -> tuple:
    """Render the forward outputs, keeping color bar and 3D at word level."""
    color_bar = make_color_bar(word_colors)
    img_2d = make_2d_image(word_colors, unit=image_unit)
    fig_3d = make_3d_tower(word_colors)
    return color_bar, img_2d, fig_3d


def forward_tab_handler(text: str, beta: float, gamma: float,
                        grain_amount: float, image_unit: str) -> tuple:
    """Gradio 탭 1 이벤트 핸들러 — 텍스트를 시각화 결과로 변환한다."""

    word_colors = run_forward(text, beta, gamma, grain_amount)
    color_bar, img_2d, fig_3d = render_forward_outputs(word_colors, image_unit)
    return word_colors, color_bar, img_2d, fig_3d


def refresh_forward_outputs(word_colors: list[dict], beta: float, gamma: float,
                            grain_amount: float, image_unit: str) -> tuple:
    """Refresh visual outputs from cached model colors without BERT inference."""
    updated = reblend_forward_results(word_colors or [], beta, gamma, grain_amount)
    return render_forward_outputs(updated, image_unit)


def refresh_forward_2d(word_colors: list[dict], beta: float, gamma: float,
                       grain_amount: float, image_unit: str):
    """Refresh only the mode/grain-sensitive 2D view."""
    updated = reblend_forward_results(word_colors or [], beta, gamma, grain_amount)
    return make_2d_image(updated, unit=image_unit)


# ─── 탭 2: 역방향 ─────────────────────────────────────────────────────────────

def make_simplified_reverse_image(pixels) -> Image.Image:
    """Enlarge the downsampled RGB grid without smoothing pixel boundaries."""
    base_img = Image.fromarray(pixels.astype("uint8"))
    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST
    return base_img.resize((512, 512), resample=resample_method)


def make_reverse_mapping_table(mapping_rows: list[dict]) -> str:
    """Build a scrollable color-to-word table for reverse output inspection."""
    body_rows = []
    for row in mapping_rows:
        rgb = row["rgb"]
        hex_color = html.escape(row["hex"])
        word = html.escape(row["word"])
        body_rows.append(
            "<tr>"
            f"<td>({row['row']}, {row['col']})</td>"
            f"<td><span class='reverse-swatch' style='background:{hex_color}'></span></td>"
            f"<td>{rgb[0]}, {rgb[1]}, {rgb[2]}</td>"
            f"<td>{hex_color}</td>"
            f"<td>{word}</td>"
            "</tr>"
        )

    return (
        "<div class='reverse-map-wrap'><table class='reverse-map-table'>"
        "<thead><tr><th>위치</th><th>색상</th><th>RGB</th><th>HEX</th><th>생성 단어</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def reverse_tab_handler(image, resolution_str: str, keyword: str, alpha: float) -> tuple:
    """Gradio 탭 2 이벤트 핸들러 — 이미지를 구조적 시로 변환한다."""

    if image is None:
        return None, "이미지를 먼저 업로드한 뒤 다시 생성해 주세요.", ""

    # 1. resolution_str("8×8")을 (H, W) 튜플로 파싱
    res_parts = resolution_str.split("×")
    resolution = (int(res_parts[0]), int(res_parts[1]))

    # 2. keyword가 비어있으면 None 처리 및 alpha=0 강제
    if not keyword or keyword.strip() == "":
        keyword = None
        alpha = 0.0

    # 3. 역방향 파이프라인 실행 및 색상-단어 매핑 결과 획득
    details = run_reverse_with_details(image, resolution, keyword, alpha)
    poem_grid = details["word_grid"]

    # 4. 2차원 행렬을 "단어 단어 단어\n단어 단어 단어\n..." 형태의 텍스트로 결합
    lines = []
    for row in poem_grid:
        lines.append(" ".join(row))

    simplified_image = make_simplified_reverse_image(details["pixels"])
    mapping_table = make_reverse_mapping_table(details["mapping_rows"])
    return simplified_image, "\n".join(lines), mapping_table


# ─── 탭 3: 순환 실험 ──────────────────────────────────────────────────────────

def cycle_tab_handler(text: str, beta: float, gamma: float,
                      resolution_str: str) -> tuple:
    """Gradio 탭 3 이벤트 핸들러 — 텍스트를 한 번에 정방향→역방향으로 순환시킨다.

    흐름: 텍스트 → run_forward → 단어별 색상 2D 이미지 → (임시 파일) →
          run_reverse → 새로운 구조적 시. 원본 텍스트와 순환 결과를 나란히 비교한다.
    """
    if not text or not text.strip():
        return None, "", "텍스트를 먼저 입력한 뒤 순환을 실행해 주세요."

    # 1. 정방향: 텍스트 → 단어별 색상. 순환에서는 단어=픽셀 1개가 직관적이라
    #    grain은 사용하지 않고 단어 단위 2D 이미지를 색상 이미지로 쓴다.
    word_colors = run_forward(text, beta, gamma, grain_amount=0.0)
    color_image = make_2d_image(word_colors, unit="word")

    # 2. 정방향 색상 이미지를 임시 파일로 저장 → 역방향 입력 경로로 전달.
    #    run_reverse는 파일 경로를 받으므로 메모리 이미지를 잠깐 디스크에 내린다.
    H, W = (int(part) for part in resolution_str.split("×"))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        color_image.save(tmp_path)

        # 3. 역방향: 색상 이미지 → 새로운 시 (키워드 없이 순수 색상 기반).
        details = run_reverse_with_details(tmp_path, (H, W), None, 0.0)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    new_poem = "\n".join(" ".join(row) for row in details["word_grid"])
    return color_image, text.strip(), new_poem


# ─── Gradio UI 구성 ───────────────────────────────────────────────────────────

def build_ui():
    # 다크 모드가 전시에 더 몰입감을 줄 수 있어 theme을 약간 어둡게 튜닝하는 것도 좋습니다.
    with gr.Blocks(title="Synesthetic AI", theme=APP_THEME, css=APP_CSS) as demo:
        gr.Markdown("<h1 style='text-align: center;'>Synesthetic AI</h1>")
        gr.Markdown("<p style='text-align: center;'>기계의 눈으로 읽고, 시의 언어로 그리는 공감각 예술</p>")

        # ─── 탭 1: 정방향 ───────────────────────────────────────────────────
        with gr.Tab("정방향: 텍스트 → 시각화"):
            forward_state = gr.State([])
            with gr.Row():
                # 좌측 (Scale=1): 깔끔한 입력 및 제어부
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="텍스트 입력 (시, 소설 등)", lines=8, placeholder="여기에 텍스트를 입력하세요...")

                    # 관람객이 복잡해하지 않도록 파라미터는 접어둡니다.
                    with gr.Accordion("⚙️ 세부 파라미터 조절", open=True):
                        beta_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="색 모델 (Cosine ←→ MLP)")
                        gamma_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="첫 글자 색 끌림")
                        grain_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="글자 그레인 강도")
                        #ncols_slider = gr.Slider(8, 32, value=32, step=1, label="2D 이미지 가로 픽셀 수")

                    # 버튼들을 나란히 배치
                    with gr.Row():
                        run_btn = gr.Button("시각화 생성", variant="primary", elem_id="generate-btn")
                        #immersive_btn = gr.Button("🚀 3D 전체화면", variant="secondary")

                # 우측 (Scale=2): 결과물 중심의 넓은 캔버스
                with gr.Column(scale=2):
                    color_bar_out = gr.Image(label="전체 색상 흐름 (Color Bar)", height=120)

                    # [수정된 부분] gr.Row()를 제거하여 위아래로 한 줄씩 큼직하게 배치합니다.
                    image_unit = gr.Radio(
                        choices=[("글자별 1픽셀", "character"), ("단어별 1픽셀", "word")],
                        value="character",
                        label="2D 이미지 단위",
                        elem_id="image-unit-toggle",
                    )
                    img_2d_out = gr.Image(label="2D 픽셀 이미지")
                    fig_3d_out = gr.Plot(label="3D 컬러 타워 (미리보기)")

            # 이벤트 연결
            #run_btn.click(forward_tab_handler, inputs=[text_input, beta_slider, gamma_slider, ncols_slider], outputs=[color_bar_out, img_2d_out, fig_3d_out])
            run_btn.click(
                forward_tab_handler,
                inputs=[text_input, beta_slider, gamma_slider, grain_slider, image_unit],
                outputs=[forward_state, color_bar_out, img_2d_out, fig_3d_out],
            )
            gr.on(
                triggers=[beta_slider.input, gamma_slider.input],
                fn=refresh_forward_outputs,
                inputs=[forward_state, beta_slider, gamma_slider, grain_slider, image_unit],
                outputs=[color_bar_out, img_2d_out, fig_3d_out],
                trigger_mode="always_last",
            )
            gr.on(
                triggers=[grain_slider.input, image_unit.change],
                fn=refresh_forward_2d,
                inputs=[forward_state, beta_slider, gamma_slider, grain_slider, image_unit],
                outputs=[img_2d_out],
                trigger_mode="always_last",
            )
            #immersive_btn.click(fn=None, inputs=None, outputs=None, js="() => { window.open('http://localhost:8000/tower_3d.html', '_blank'); }")
            #text_input.change(fn=update_ncols_max, inputs=text_input, outputs=ncols_slider)
        # ─── 탭 2: 역방향 ───────────────────────────────────────────────────
        with gr.Tab("역방향: 이미지 → 시"):
            with gr.Row(elem_id="reverse-layout"):
                # 좌측 (Scale=1)
                with gr.Column(scale=1, min_width=280):
                    image_input = gr.Image(type="filepath", label="영감을 줄 이미지 업로드")

                    with gr.Accordion("⚙️ 시 생성 옵션", open=False):
                        resolution_dropdown = gr.Dropdown(["8×8", "10×10", "16×16", "32×32"], value="10×10", label="추상화 해상도", elem_id="resolution-dropdown")
                        keyword_input = gr.Textbox(label="심상 키워드 (선택)")
                        alpha_slider = gr.Slider(0.0, 1.0, value=0.5, label="α (순수 색상 ←→ 키워드 문맥)")

                    reverse_btn = gr.Button("✍️ 공감각적 시 생성", variant="primary", elem_id="reverse-generate-btn")

                # 우측 (Scale=2)
                with gr.Column(scale=2, min_width=320, elem_id="reverse-results-column"):
                    simplified_image_out = gr.Image(label="단순화된 색상 이미지")
                    poem_out = gr.Textbox(label="생성된 구조적 시", lines=20)
                    gr.Markdown("### 색상과 생성 단어의 대응")
                    mapping_table_out = gr.HTML()

            # 이벤트 연결
            reverse_btn.click(
                reverse_tab_handler,
                inputs=[image_input, resolution_dropdown, keyword_input, alpha_slider],
                outputs=[simplified_image_out, poem_out, mapping_table_out],
            )

        # ─── 탭 3: 순환 실험 ────────────────────────────────────────────────
        with gr.Tab("순환 실험: 텍스트 → 색 → 시"):
            gr.Markdown(
                "텍스트를 색으로 바꾸고(정방향), 그 색을 다시 시로 되돌립니다(역방향). "
                "원본과 순환 결과를 나란히 비교해 모델이 의미↔색을 얼마나 잘 번역하는지 봅니다."
            )
            with gr.Row():
                # 좌측 (Scale=1): 입력 및 제어
                with gr.Column(scale=1, min_width=280):
                    cycle_text_input = gr.Textbox(
                        label="원본 텍스트 (시, 문장 등)", lines=8,
                        placeholder="순환시킬 텍스트를 입력하세요...",
                    )
                    with gr.Accordion("⚙️ 순환 옵션", open=True):
                        cycle_beta = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                               label="색 모델 (Cosine ←→ MLP)")
                        cycle_gamma = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                                label="첫 글자 색 끌림")
                        cycle_resolution = gr.Dropdown(
                            ["8×8", "10×10", "16×16", "32×32"], value="10×10",
                            label="역방향 추상화 해상도", elem_id="resolution-dropdown",
                        )
                    cycle_btn = gr.Button("🔄 순환 실행", variant="primary",
                                          elem_id="generate-btn")

                # 우측 (Scale=2): 원본 시 | 색상 이미지 | 생성된 시 3단 비교
                with gr.Column(scale=2):
                    with gr.Row():
                        cycle_original_out = gr.Textbox(
                            label="① 원본 시", lines=18, interactive=False)
                        cycle_image_out = gr.Image(label="② 색상 이미지 (정방향)")
                        cycle_new_poem_out = gr.Textbox(
                            label="③ 순환 후 생성된 시 (역방향)", lines=18,
                            interactive=False)

            cycle_btn.click(
                cycle_tab_handler,
                inputs=[cycle_text_input, cycle_beta, cycle_gamma, cycle_resolution],
                outputs=[cycle_image_out, cycle_original_out, cycle_new_poem_out],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # share=True를 주면 외부 링크가 생성되어 교수님/팀원 시연에 편리합니다.
    demo.launch(share=True)
