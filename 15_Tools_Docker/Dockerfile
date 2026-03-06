FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 로컬 프로젝트 구조를 컨테이너에도 동일하게 복사
COPY . /app

# main.py가 os.getcwd() 기준으로 경로를 계산하므로 notebooks에서 실행
WORKDIR /app/notebooks
CMD ["python", "main.py"]