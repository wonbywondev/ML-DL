# 영화 정보 서비스

TMDB 인기 영화 30개를 기반으로 영화를 조회하고 관리할 수 있는 웹 서비스입니다.

## Streamlit Cloud에서 사용하기

별도 설치 없이 아래 링크에서 바로 사용할 수 있습니다.

- 서비스 URL: https://nqcnn7mmygkfrdw9n54mvx.streamlit.app/

> 데이터가 없을 경우 상단 **🔄** 버튼으로 TMDB에서 영화 데이터를 수집할 수 있습니다.
> TMDB 토큰은 서버에 설정되어 있습니다.

## 로컬에서 실행하기

### 사전 준비

TMDB API 토큰을 발급받아 `.env` 파일을 생성합니다.

```
TMDB_TOKEN=your_tmdb_token_here
```

> TMDB 토큰은 [themoviedb.org](https://www.themoviedb.org/) 회원가입 후 무료로 발급받을 수 있습니다.

### 의존성 설치

**uv**
```bash
uv sync
```

**pip**
```bash
pip install -r requirements.txt
```

### 백엔드 실행

**uv**
```bash
uv run uvicorn apps.backend:app --reload
```

**pip**
```bash
uvicorn apps.backend:app --reload
```

### 프론트엔드 실행

**uv**
```bash
uv run streamlit run apps/streamlit.py
```

**pip**
```bash
streamlit run apps/streamlit.py
```

브라우저에서 `http://localhost:8501` 접속

## 사용 방법

### 영화 목록 보기
접속하면 TMDB 인기 영화 30개가 포스터와 함께 표시됩니다.
데이터가 없을 경우 상단 **🔄** 버튼으로 새로 수집할 수 있습니다.

### 영화 등록
페이지 상단 폼에 제목, 개봉일, 감독, 장르, 포스터 URL을 입력 후 등록합니다.

### 영화 삭제
우측 상단 **🛠 관리** 버튼을 누르면 각 카드에 삭제 버튼이 나타납니다.