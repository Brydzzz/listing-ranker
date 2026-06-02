import csv
import random
from pathlib import Path

from fastapi import APIRouter

from app.schemas import Listing

LISTINGS_CSV = Path("listings4.csv")

router = APIRouter()


@router.get("/listings", response_model=list[Listing])
def get_listings():
    with open(LISTINGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    session_ids = list({row["session_id"] for row in rows})
    chosen_session_id = random.choice(session_ids)

    listings = [
        Listing(
            id=int(row["id"]),
            name=row["name"],
            price=float(row["price"]),
            rating=float(row["rating"]),
            number_of_reviews=int(row["number_of_reviews"]),
            neighbourhood=row["neighbourhood"],
            session_id=row["session_id"],
        )
        for row in rows
        if row["session_id"] == chosen_session_id
    ]
    return listings
