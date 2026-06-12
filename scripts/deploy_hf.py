"""
deploy_hf.py — Synesthetic AI 를 Hugging Face Spaces(Gradio SDK)로 배포한다.

전제:
    - 사전에 `hf auth login` 으로 HF 토큰이 로컬 캐시에 저장돼 있어야 한다.
    - data/candidate_vectors.npy, data/mlp_weights.pt 가 로컬에 존재해야 한다
      (.gitignore 대상이라 GitHub 엔 없지만 HF Space 런타임엔 반드시 필요).

동작:
    1. 로그인 사용자명으로 Space repo(<user>/synesthetic-ai)를 생성(있으면 재사용)
    2. HF 전용 README.md(YAML front matter 포함)를 동적 생성해 업로드
       → 로컬/GitHub README.md 는 건드리지 않는다
    3. 런타임에 필요한 파일만 선별 업로드(대용량 .npy 는 hub 가 LFS 자동 처리)

실행:
    python scripts/deploy_hf.py
"""

import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "synesthetic-ai"          # URL slug (소문자/하이픈)
SPACE_TITLE = "Synesthetic AI"         # 표시 이름 = 프로젝트 이름
SDK_VERSION = "6.14.0"                 # launch(theme=, css=) 동작에 gradio 6.x 필요

# HF Space 카드용 front matter + 간단 소개. (GitHub README 와 별개)
README_FRONT_MATTER = f"""---
title: {SPACE_TITLE}
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: {SDK_VERSION}
app_file: src/app.py
pinned: false
license: mit
short_description: 단어에서 색으로, 색에서 시로 — 공감각 AI 탐구
---

# {SPACE_TITLE}

단어에서 색으로, 색에서 시로 — 텍스트만 학습한 BERT 임베딩 공간에
인간 공감각자(synesthete)와 유사한 색채 구조가 내재되어 있는지 탐구하는 데모입니다.

- **정방향**: 텍스트 → 단어별 RGB(색상 바·2D 픽셀·3D 타워·의미 공간)
- **역방향**: 이미지 → 색을 시어로 번역한 구조적 시
- **순환 실험**: 텍스트 → 색 → 시 왕복 비교

> 첫 기동 시 BERT(bert-base-uncased) 다운로드와 모델 로드로 수 분이 걸릴 수 있습니다.
"""

# 런타임에 실제로 필요한 파일만 업로드한다(학습 전용 캐시·연구용 원자료 제외).
ALLOW_PATTERNS = [
    "src/*.py",
    "requirements.txt",
    "data/bert_mean_vec.npy",
    "data/bert_anchor_R.npy",
    "data/bert_anchor_G.npy",
    "data/bert_anchor_B.npy",
    "data/synesthesia_grapheme_mean_rgb.csv",
    "data/nrc_word_rgb.csv",
    "data/poetry_candidate_words.txt",
    "data/candidate_vectors.npy",   # ~29MB, hub 가 LFS 자동 처리
    "data/mlp_weights.pt",          # 99KB
]


def _check_required_files() -> None:
    required = [
        "data/candidate_vectors.npy",
        "data/mlp_weights.pt",
        "data/bert_anchor_R.npy",
        "src/app.py",
    ]
    missing = [p for p in required if not (REPO_ROOT / p).exists()]
    if missing:
        print("[deploy] 필수 파일 누락 — 업로드 중단:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def main() -> None:
    _check_required_files()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"[deploy] 대상 Space: {repo_id}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    # HF 전용 README(front matter) 업로드 — 로컬 README.md 는 그대로 둔다.
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as f:
        f.write(README_FRONT_MATTER)
        readme_tmp = f.name
    api.upload_file(
        path_or_fileobj=readme_tmp,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add Space card (Synesthetic AI)",
    )

    print("[deploy] 파일 업로드 중(대용량 벡터는 LFS 자동 처리)...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=ALLOW_PATTERNS,
        commit_message="Deploy Synesthetic AI app + runtime data",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n[deploy] ✅ 업로드 완료. 빌드가 끝나면 아래 링크로 24/7 접근 가능합니다:")
    print(f"  {url}")
    print("  (첫 빌드: 의존성 설치 + BERT 다운로드로 수 분 소요)")


if __name__ == "__main__":
    main()
