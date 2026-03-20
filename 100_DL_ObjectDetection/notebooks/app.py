import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Webcam YOLO BBox Check", layout="wide")
st.title("📷 웹캠 → YOLO 추론 (스냅샷)")
st.caption("버튼 눌러 촬영 → bbox가 잘 잡히는지 확인")

# ✅ 네 모델 경로로 바꿔
MODEL_PATH = "best(fit_s3).pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

conf = st.sidebar.slider("confidence", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("imgsz", [320, 480, 640, 960], index=2)

shot = st.camera_input("알약을 카메라에 보여주고 촬영해줘")

if shot is not None:
    # bytes -> numpy(BGR)
    img = Image.open(shot).convert("RGB")
    img_np = np.array(img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # YOLO 추론
    results = model.predict(source=bgr, conf=conf, imgsz=imgsz, verbose=False)

    # 결과 그려진 이미지 (ultralytics 내장 plot)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(img_np, use_container_width=True)
    with col2:
        st.subheader("추론 결과(bbox)")
        st.image(annotated_rgb, use_container_width=True)

    # 디텍션 요약
    boxes = results[0].boxes
    st.write(f"탐지 수: **{len(boxes)}**")
    if len(boxes) > 0:
        # cls/conf/xyxy 테이블
        rows = []
        for b in boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            rows.append({"cls": cls_id, "score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        st.dataframe(rows, use_container_width=True)
