import os
import urllib.request

import altair as alt
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "mnist-12"
MODEL_PATH = f"models/{MODEL_NAME}/{MODEL_NAME}.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/"
    "vision/classification/mnist/model/mnist-12.onnx"
)
CANVAS_SIZE = 280  # 캔버스 픽셀 크기 (28x28의 10배)
INPUT_SIZE = 28    # 모델 입력 크기


# ── 모델 관리 ──────────────────────────────────────────────────────────────────
def download_model(url: str, path: str) -> None:
    """모델 파일이 없을 경우 GitHub에서 다운로드"""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with st.spinner("MNIST 모델 다운로드 중..."):
        urllib.request.urlretrieve(url, path)


@st.cache_resource
def load_model(model_path: str) -> ort.InferenceSession:
    """ONNX 모델 로드 (모델 경로를 캐시 키로 사용)"""
    download_model(MODEL_URL, model_path)
    return ort.InferenceSession(model_path)


# ── 이미지 처리 ────────────────────────────────────────────────────────────────
def preprocess(canvas_image: np.ndarray) -> tuple[np.ndarray, Image.Image]:
    """
    캔버스 RGBA 이미지를 MNIST 모델 입력 형식으로 전처리.

    Returns:
        input_tensor: shape (1, 1, 28, 28), float32
        preview_img: 전처리 결과를 시각화한 PIL 이미지
    """
    # RGBA → 그레이스케일 반전 (흰 배경에 검정 획 → MNIST는 검정 배경에 흰 글씨)
    gray = (255 - canvas_image[:, :, 0]).astype(np.float32)

    # 28x28 리사이즈
    pil_img = Image.fromarray(gray.astype(np.uint8)).resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    # 0~255 → 0~1 정규화
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    # (1, 1, 28, 28) NCHW 형태로 변환
    input_tensor = img_array.reshape(1, 1, INPUT_SIZE, INPUT_SIZE)

    return input_tensor, pil_img


def run_inference(session: ort.InferenceSession, input_tensor: np.ndarray) -> np.ndarray:
    """ONNX 모델 추론 후 softmax 확률 반환"""
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0][0]

    # softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs


# ── 메인 앱 ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST 숫자 인식", layout="wide")
st.title("✏️ 손글씨 숫자 인식 (MNIST)")


session = load_model(MODEL_PATH)

# 이미지 저장소 초기화
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # list of (PIL.Image, label, prob)

# undo/redo 히스토리 초기화
EMPTY_DRAWING = {"version": "4.4.0", "objects": []}
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []
if "current_drawing" not in st.session_state:
    st.session_state.current_drawing = EMPTY_DRAWING
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# ── 1. 입력 캔버스 / 2. 전처리 이미지 ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("입력 캔버스")

    # undo/redo/reset 버튼 (캔버스보다 먼저 렌더링하여 스택 상태 반영)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("↩️ Undo", use_container_width=True, disabled=len(st.session_state.undo_stack) == 0):
            st.session_state.redo_stack.append(st.session_state.current_drawing)
            st.session_state.current_drawing = st.session_state.undo_stack.pop()
            st.session_state.canvas_key += 1
            st.rerun()
    with btn_col2:
        if st.button("↪️ Redo", use_container_width=True, disabled=len(st.session_state.redo_stack) == 0):
            st.session_state.undo_stack.append(st.session_state.current_drawing)
            st.session_state.current_drawing = st.session_state.redo_stack.pop()
            st.session_state.canvas_key += 1
            st.rerun()
    with btn_col3:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.undo_stack.append(st.session_state.current_drawing)
            st.session_state.redo_stack.clear()
            st.session_state.current_drawing = EMPTY_DRAWING
            st.session_state.canvas_key += 1
            st.rerun()

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        display_toolbar=False,
        initial_drawing=st.session_state.current_drawing,
        key=f"canvas_{st.session_state.canvas_key}",
    )

with col2:
    st.subheader("전처리 이미지 (28×28)")
    preview_placeholder = st.empty()

# ── 3. 추론 결과 ───────────────────────────────────────────────────────────────
st.subheader("모델 추론 결과")
chart_placeholder = st.empty()

# ── 캔버스 히스토리 업데이트 ────────────────────────────────────────────────────
if canvas_result.json_data is not None:
    new_drawing = canvas_result.json_data
    new_count = len(new_drawing.get("objects", []))
    cur_count = len(st.session_state.current_drawing.get("objects", []))
    # 스트로크가 추가됐을 때만 undo 스택에 push 후 rerun으로 버튼 상태 갱신
    if new_count > cur_count:
        st.session_state.undo_stack.append(st.session_state.current_drawing)
        st.session_state.redo_stack.clear()
        st.session_state.current_drawing = new_drawing
        st.rerun()

# ── 캔버스 입력 처리 ───────────────────────────────────────────────────────────
has_drawing = (
    canvas_result.json_data is not None
    and len(canvas_result.json_data.get("objects", [])) > 0
)
if canvas_result.image_data is not None and has_drawing:
    img_data = canvas_result.image_data
    input_tensor, preview_img = preprocess(img_data)
    probs = run_inference(session, input_tensor)

    top3_idx = probs.argsort()[::-1][:3]
    label = int(top3_idx[0])
    confidence = float(probs[label])

    caption = "  |  ".join(
        f"{'1위' if i == 0 else f'{i+1}위'}: {top3_idx[i]} ({probs[top3_idx[i]]:.1%})"
        for i in range(3)
    )

    # 전처리 이미지 표시
    preview_placeholder.image(
        preview_img,
        caption=caption,
        width=CANVAS_SIZE,
    )

    # bar chart (y축 0~1 고정)
    df = pd.DataFrame({"숫자": [str(i) for i in range(10)], "확률": probs.tolist()})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("숫자:N", sort=None),
            y=alt.Y("확률:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum["숫자"] == str(label),
                alt.value("#f63366"),
                alt.value("#4C78A8"),
            ),
        )
        .properties(height=300)
    )
    chart_placeholder.altair_chart(chart, use_container_width=True)

    # 저장 버튼
    if st.button("이미지 저장"):
        st.session_state.gallery.append((preview_img.copy(), label, confidence))
        st.success(f"저장 완료: 예측 {label} ({confidence:.1%})")
else:
    empty_chart = (
        alt.Chart(pd.DataFrame({"숫자": [str(i) for i in range(10)], "확률": [0.0] * 10}))
        .mark_bar()
        .encode(
            x=alt.X("숫자:N", sort=None),
            y=alt.Y("확률:Q", scale=alt.Scale(domain=[0, 1])),
        )
        .properties(height=300, title="이미지를 그려주세요.")
    )
    chart_placeholder.altair_chart(empty_chart, use_container_width=True)

# ── 4. 이미지 저장소 ───────────────────────────────────────────────────────────
st.subheader("이미지 저장소")

if st.session_state.gallery:
    cols = st.columns(5)
    for i, (img, lbl, prob) in enumerate(reversed(st.session_state.gallery)):
        with cols[i % 5]:
            st.image(img, caption=f"{lbl}  ({prob:.1%})", width=100)
else:
    st.caption("저장된 이미지가 없습니다. 숫자를 그린 후 '이미지 저장' 버튼을 누르세요.")
