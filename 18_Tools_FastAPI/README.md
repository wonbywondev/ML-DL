# 영화 정보 서비스

TMDB 기반 영화 데이터를 제공하는 웹 애플리케이션입니다.
**FastAPI** 백엔드와 **Streamlit** 프론트엔드로 구성됩니다.

## 서비스 구조

```
Streamlit (frontend) ──HTTP──▶ FastAPI (backend) ──▶ movies.json (DB)
                                        ▲
                               TMDB API (crawler)
```

## 기능

- 영화 목록 조회 (포스터, 제목, 장르, 감독, 개봉일)
- 영화 직접 등록 / 삭제
- 관리 모드 토글 (삭제 버튼 노출 제어)
- TMDB 크롤러로 인기 영화 30개 시드 데이터 수집

## TMDB 크롤러 (`apps/crawler.py`)

[TMDB API](https://www.themoviedb.org/documentation/api)를 사용해 인기 영화 데이터를 수집합니다.

- `popular` 엔드포인트로 영화 목록 수집
- `credits` 엔드포인트로 감독 정보 별도 조회
- 수집 항목: 제목, 개봉일, 감독, 장르, 포스터 URL
- 결과를 `data/movies.json`에 저장 (백엔드 시드 데이터로 자동 로드)

TMDB API 키는 `.env` 파일에 `TMDB_TOKEN`으로 설정합니다.

```bash
# 재수집 시
uv run python -m apps.crawler
```

## 심화 기능 (미구현)

아래 기능은 미션 시나리오의 선택 과제로, 현재 버전에서는 구현하지 않았습니다.

| 기능 | 설명 |
|---|---|
| 리뷰 등록/조회/삭제 | 영화별 사용자 리뷰 관리 API |
| 감성 분석 | 리뷰 작성 시 한국어 감성 분석 모델 자동 실행 |
| 평균 평점 | 감성 분석 점수 기반 영화별 평균 평점 산출 |
| 리뷰 목록 표시 | 최근 10개 리뷰 (작성일, 내용, 감성 결과) 표시 |

## 실행 방법

### 사전 준비

```bash
# 의존성 설치
uv sync

# .env 파일 생성
echo "TMDB_TOKEN=your_token_here" > .env
```

### 백엔드 실행

```bash
uv run uvicorn apps.backend:app --reload
# http://localhost:8000/docs 에서 API 문서 확인
```

### 프론트엔드 실행

```bash
uv run streamlit run apps/streamlit.py
# http://localhost:8501
```

### 시드 데이터 재수집

```bash
uv run python -m apps.crawler
```

### 테스트 실행

```bash
uv run pytest tests/ -v
```

## 프로젝트 구조

```
18_Tools_FastAPI/
├── apps/
│   ├── backend.py      # FastAPI 서버
│   ├── streamlit.py    # Streamlit 프론트엔드
│   └── crawler.py      # TMDB 크롤러
├── data/
│   └── movies.json     # 영화 데이터 (파일 기반 DB)
├── tests/
│   └── test_backend.py # pytest 테스트
└── docs/
    └── scenario.md     # 미션 시나리오
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/movies` | 전체 영화 조회 |
| GET | `/movies/{id}` | 특정 영화 조회 |
| POST | `/movies` | 영화 등록 |
| DELETE | `/movies/{id}` | 영화 삭제 |
