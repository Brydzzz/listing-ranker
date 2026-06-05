import os
from fastapi.testclient import TestClient

from app.ab_router import get_model_for_user
from main import app

if os.path.exists("results.csv"):
    os.remove("results.csv")

client = TestClient(app)

response = client.get("/listings")
if response.status_code != 200:
    exit(1)

listings = response.json()

users = ["user-4", "user-1"]

for user in users:
    model, model_name = get_model_for_user(user)

    response = client.post("/rank", json=listings, headers={"X-User-Id": user})
    body = response.json()
    
    ranked_listings = body.get('ranked_listings', [])
    model_used = body.get("model_used")
    
    events_payload = []
    for i, lst in enumerate(ranked_listings):
        if user == "user-1":
            if i == 0:
                s_type = "book_listing"
            elif i < 5:
                s_type = "view_listing"
            else:
                s_type = "not_viewed"
        else:
            if i < 3:
                s_type = "view_listing"
            else:
                s_type = "not_viewed"
                
        events_payload.append({
            "user_uid": user,
            "session_type": s_type,
            "ranking_model": model_used,
            "listing": lst,
        })
        
    client.post("/session-event", json=events_payload)
