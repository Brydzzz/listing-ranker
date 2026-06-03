from fastapi import FastAPI

from app.listings_router import router as listings_router
from app.rank_router import router as rank_router
from app.sessions_router import router as sessions_router

app = FastAPI(title="Listing Ranking Microservice")
app.include_router(listings_router)
app.include_router(rank_router)
app.include_router(sessions_router)
