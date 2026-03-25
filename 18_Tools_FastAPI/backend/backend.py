import json
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from datetime import date

app = FastAPI()


DB_PATH = Path(__file__).parent.parent / "data" / "movies.json"


def _load() -> tuple[dict[int, dict], int]:
    if not DB_PATH.exists():
        return {}, 1
    raw = json.loads(DB_PATH.read_text())
    db = {i + 1: m for i, m in enumerate(raw)}
    return db, len(db) + 1


def _save():
    DB_PATH.parent.mkdir(exist_ok=True)
    DB_PATH.write_text(json.dumps(list(movies.values()), ensure_ascii=False, indent=2))


# In-memory DB
movies, next_id = _load()


class MovieCreate(BaseModel):
    title: str
    release_date: date
    director: str
    genre: str
    poster_url: str


class Movie(MovieCreate):
    id: int


# 전체 영화 조회
@app.get("/movies", response_model=list[Movie])
def get_movies():
    return [Movie(id=k, **v) for k, v in movies.items()]


# 특정 영화 조회
@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie(movie_id: int):
    if movie_id not in movies:
        raise HTTPException(status_code=404, detail="Movie not found")
    return Movie(id=movie_id, **movies[movie_id])


# 영화 등록
@app.post("/movies", response_model=Movie, status_code=201)
def add_movie(movie: MovieCreate):
    global next_id
    movies[next_id] = movie.model_dump(mode="json")
    created = Movie(id=next_id, **movies[next_id])
    next_id += 1
    _save()
    return created


# 영화 삭제
@app.delete("/movies/{movie_id}", status_code=204)
def remove_movie(movie_id: int):
    if movie_id not in movies:
        raise HTTPException(status_code=404, detail="Movie not found")
    del movies[movie_id]
    _save()


# TMDB 크롤링
@app.post("/crawl", status_code=200)
def crawl_movies():
    global movies, next_id
    from backend.crawler import crawl
    result = crawl()
    movies = {i + 1: m for i, m in enumerate(result)}
    next_id = len(movies) + 1
    _save()
    return {"count": len(movies)}
