import json
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from datetime import date

app = FastAPI()


def _load_seed() -> tuple[dict[int, dict], int]:
    seed_path = Path(__file__).parent.parent / "data" / "movies.json"
    if not seed_path.exists():
        return {}, 1
    raw = json.loads(seed_path.read_text())
    db = {i + 1: m for i, m in enumerate(raw)}
    return db, len(db) + 1


# In-memory DB
movies, next_id = _load_seed()


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
    movies[next_id] = movie.model_dump()
    created = Movie(id=next_id, **movies[next_id])
    next_id += 1
    return created


# 영화 삭제
@app.delete("/movies/{movie_id}", status_code=204)
def remove_movie(movie_id: int):
    if movie_id not in movies:
        raise HTTPException(status_code=404, detail="Movie not found")
    del movies[movie_id]
