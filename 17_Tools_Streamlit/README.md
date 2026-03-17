# MNIST 손글씨 숫자 인식 서비스

웹 브라우저에서 마우스로 숫자를 그리면 MNIST ONNX 모델이 실시간으로 숫자를 예측하는 Streamlit 기반 웹 서비스입니다.

## 실행 방법

```bash
docker run -p 8501:8501 pixar12372/mnist-streamlit:latest
```

브라우저에서 [http://localhost:8501](http://localhost:8501) 접속

## 화면 구성

| 영역 | 설명 |
|------|------|
| 입력 캔버스 | 마우스로 숫자를 그리는 280×280 캔버스. Undo / Redo / Reset 버튼 포함 |
| 전처리 이미지 | 캔버스 입력을 28×28 흑백으로 전처리한 결과 + top3 예측 캡션 |
| 모델 추론 결과 | 0~9 각 숫자의 예측 확률 bar chart (y축 0~1 고정) |

## Docker Hub

[https://hub.docker.com/r/pixar12372/mnist-streamlit](https://hub.docker.com/r/pixar12372/mnist-streamlit)

## 로컬 개발 환경

<details>
<summary>uv</summary>

```bash
uv sync
uv run streamlit run app.py
```

</details>

<details>
<summary>pip</summary>

```bash
pip install streamlit streamlit-drawable-canvas onnxruntime pillow numpy altair
streamlit run app.py
```

</details>

> 로컬 실행 시 앱 시작 시 모델 파일이 없으면 자동으로 다운로드됩니다.
