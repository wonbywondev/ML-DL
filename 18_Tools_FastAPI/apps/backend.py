from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from datetime import date

app = FastAPI()

# In-memory DB
movies: dict[int, dict] = {}
next_id: int = 1


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
