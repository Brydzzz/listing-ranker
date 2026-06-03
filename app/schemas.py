from pydantic import BaseModel


class Listing(BaseModel):
    id: int
    name: str
    price: float
    rating: float
    number_of_reviews: int
    neighbourhood: str
    session_id: str
