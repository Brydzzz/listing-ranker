import csv
from enum import Enum
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.schemas import Listing

RESULTS_CSV = Path("results.csv")

router = APIRouter()


class SessionType(str, Enum):
    view_listing = "view_listing"
    book_listing = "book_listing"
    not_viewed = "not_viewed"


class RankingModel(str, Enum):
    model1 = "model1"
    model2 = "model2"


class SessionEvent(BaseModel):
    user_uid: str
    session_type: SessionType
    ranking_model: RankingModel
    listing: Listing


@router.post("/session-event")
def record_session_event(events: list[SessionEvent]):
    file_exists = RESULTS_CSV.exists()
    
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = None
        for event in events:
            listing_dict = event.listing.model_dump()
            row = {"user_uid": event.user_uid, "session_type": event.session_type.value, "ranking_model": event.ranking_model.value, **listing_dict}
            
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
            
            writer.writerow(row)

    return {"status": "ok"}
