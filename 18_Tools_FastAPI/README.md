# 영화 정보 서비스

TMDB 인기 영화 30개를 기반으로 영화를 조회하고 관리할 수 있는 웹 서비스입니다.

## 시작하기

### 1. 의존성 설치

**uv 사용 시**
```bash
uv sync
```

**pip 사용 시**
```bash
pip install -r requirements.txt
```

### 2. 백엔드 실행

**uv 사용 시**
```bash
uv run uvicorn apps.backend:app --reload
```

**pip 사용 시**
```bash
uvicorn apps.backend:app --reload
```

### 3. 프론트엔드 실행

**uv 사용 시**
```bash
uv run streamlit run apps/streamlit.py
```

**pip 사용 시**
```bash
streamlit run apps/streamlit.py
```

브라우저에서 `http://localhost:8501` 접속

## 사용 방법

### 영화 목록 보기
접속하면 TMDB 인기 영화 30개가 포스터와 함께 표시됩니다.

### 영화 등록
페이지 상단 폼에 제목, 개봉일, 감독, 장르, 포스터 URL을 입력 후 등록합니다.

### 영화 삭제
우측 상단 **🛠 관리** 버튼을 누르면 각 카드에 삭제 버튼이 나타납니다.

## 배포된 서비스

- 프론트엔드: Streamlit Cloud
- 백엔드: Railway

## 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `TMDB_TOKEN` | TMDB API 읽기 액세스 토큰 | 없음 (필수) |
| `BACKEND_URL` | FastAPI 백엔드 주소 | `http://localhost:8000` |
