from fastapi import APIRouter, Header
from pydantic import BaseModel

from app.ab_router import get_model_for_user
from app.schemas import Listing

router = APIRouter()


class RankResponse(BaseModel):
    model_used: str
    ranked_listings: list[Listing]


@router.post("/rank", response_model=RankResponse)
def rank_listings(
    listings: list[Listing],
    x_user_id: str = Header(...),
):
    model, model_name = get_model_for_user(x_user_id)
    ranked = model.rank(listings)
    return RankResponse(model_used=model_name, ranked_listings=ranked)
