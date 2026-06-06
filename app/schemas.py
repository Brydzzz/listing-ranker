from pydantic import BaseModel


class Listing(BaseModel):
    listing_id: int
    session_id: str
    host_since: str
    host_response_rate: float
    host_acceptance_rate: float
    host_is_superhost: bool
    host_has_profile_pic: bool
    host_identity_verified: bool
    bathrooms: int
    bedrooms: int
    beds: int
    price: float
    review_scores_rating: float
    review_scores_accuracy: float
    review_scores_cleanliness: float
    review_scores_checkin: float
    review_scores_communication: float
    review_scores_location: float
    review_scores_value: float
    license: str
    instant_bookable: bool
    host_response_time: str
    property_type: str
    room_type: str
    amenities: list[str]
