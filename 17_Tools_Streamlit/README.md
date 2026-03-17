# MNIST 손글씨 숫자 인식 서비스

웹 브라우저에서 마우스로 숫자를 그리면 MNIST ONNX 모델이 실시간으로 숫자를 예측하는 Streamlit 기반 웹 서비스입니다.

## 화면 구성

| 영역 | 설명 |
|------|------|
| 입력 캔버스 | 마우스로 숫자를 그리는 280×280 캔버스. Undo / Redo / Reset 버튼 포함 |
| 전처리 이미지 | 캔버스 입력을 28×28 흑백으로 전처리한 결과 + top3 예측 캡션 |
| 모델 추론 결과 | 0~9 각 숫자의 예측 확률 bar chart (y축 0~1 고정) |

## 실행 방법

### 로컬 실행 (uv)

```bash
# 의존성 설치
uv sync

# 앱 실행
uv run streamlit run app.py
```

앱 시작 시 모델 파일(`models/mnist-12/mnist-12.onnx`)이 없으면 자동으로 다운로드됩니다.

### Docker 실행

```bash
docker run -p 8501:8501 pixar12372/mnist-streamlit:latest
```

브라우저에서 [http://localhost:8501](http://localhost:8501) 접속

### Docker 직접 빌드

```bash
docker build -t mnist-streamlit .
docker run -p 8501:8501 mnist-streamlit
```

## 프로젝트 구조

```
17_Tools_Streamlit/
├── app.py              # Streamlit 앱 메인
├── Dockerfile
├── pyproject.toml
├── models/
│   └── mnist-12/
│       └── mnist-12.onnx   # 앱 시작 시 자동 다운로드
├── notebooks/
│   └── preparation.ipynb   # 전처리 및 모델 테스트
└── docs/
    ├── scenario.md         # 미션 요구사항
    └── report.md           # 구현 보고서 (디버깅 기록 포함)
```

## 기술 스택

- Python 3.12
- Streamlit + streamlit-drawable-canvas
- ONNX Runtime (`mnist-12.onnx`, float32)
- Pillow / NumPy / Altair

## Docker Hub

```
pixar12372/mnist-streamlit:latest
```

[https://hub.docker.com/r/pixar12372/mnist-streamlit](https://hub.docker.com/r/pixar12372/mnist-streamlit)
