FROM python:3.12-slim

WORKDIR /app

# 의존성 설치
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    altair>=6.0.0 \
    numpy \
    onnxruntime \
    pillow \
    streamlit \
    streamlit-drawable-canvas

# 소스 복사
COPY app.py .
COPY models/ models/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
