# Listing Ranker - rental listing ranking with A/B testing

A LightGBM model that ranks short-term rental listings. The project contains a data preparation pipeline, model experiments, a FastAPI microservice that assigns a model to each user (A/B test), and an analysis of the test results.

> **Note:** This project was created as part of a university course. It simulated working with a client who needs a machine learning solution: from analyzing the business problem and the data, through building and comparing models, to deploying the model in a service and verifying the business criterion with an A/B test.

## Repository structure

```sh
.
├── ab_analysis.ipynb # notebook analyzing the experiment results
├── app # microservice files
│   ├── example_flow.py # example usage of the microservice
│   ├── __init__.py
│   ├── listings_router.py # APIRouter for listings
│   ├── model_selector.py # consistent hashing implementation for assigning a model
│   ├── models.py # inference for the two models: baseline and target
│   ├── rank_router.py # APIRouter returning a ranked list of listings
│   ├── schemas.py # Listing schema
│   └── sessions_router.py # APIRouter for logging user session events
├── column_embedding.py # script creating embeddings of name and description
├── combine_data.py # script that joins and transforms the data
├── extract_ab_data.py # splits the output of combine_data into A/B test data and the rest
├── main.py # FastAPI app, run with `uvicorn main:app`
├── pyproject.toml
├── README.md
├── run_lgbm_experiments_amenitites_on.py # runs configurations with different amenities handling
├── run_lgbm_experiments_params.py # runs configurations with different LightGBM parameters
├── run_lgbm_experiments.py # runs configurations with different feature sets
├── single_lgbm_run.py # a single model training run
├── simple_model_only_tf.py # early prototype: TensorFlow Ranking model (not used by the microservice)
├── simulate_ab.py # simulates a larger amount of A/B test data
└── uv.lock
```

## Files produced and required by the scripts

```sh
.
├── combined_ab.parquet # output of extract_ab_data.py, used by the microservice
├── combined_no_ab.parquet # output of extract_ab_data.py, used by single_lgbm_run.py
├── combined.parquet # output of combine_data.py
├── embeddings.npz # output of column_embedding.py, used by single_lgbm_run.py
├── eval_dataset.bin # created by single_lgbm_run, allows re-running training without rebuilding the lgb.Dataset
├── lgbm_model_with_feature_names.txt # saved model
├── listing_ids.npy # output of column_embedding.py, used by single_lgbm_run.py
├── metadata_cache.pkl # created by single_lgbm_run, allows re-running training without rebuilding the lgb.Dataset
├── results.csv # microservice results
├── listings.csv # required to build combined.parquet and the embeddings
├── sessions.csv # required to build combined.parquet
├── test_data_cache.parquet # created by single_lgbm_run, allows re-running training without rebuilding the lgb.Dataset
└── train_dataset.bin # created by single_lgbm_run, allows re-running training without rebuilding the lgb.Dataset
```

## About the microservice

- The `/listings` endpoint simulates listings returned by a search engine

- The `/session-event` endpoint logs A/B test results for later analysis.

- Each user is assigned a specific model using a hash function.

- `example_flow.py` shows an example of using the microservice.

- `simulate_ab.py` generates synthetic A/B test results to be analyzed by `ab_analysis.ipynb`.

- `ab_analysis.ipynb` verifies the business criterion by comparing average session length.

### Usage example - `curl` and analysis in a Jupyter notebook

#### Starting the microservice

First start the microservice: `uvicorn main:app`
```bash
❯ uvicorn main:app
INFO:     Started server process [7785]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### Getting a list of listings simulating search results

```bash
❯ curl -s http://localhost:8000/listings
```

```json
# response
[
    {
        "listing_id":1137096251242024922,
        "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
        "host_since":"2023-10-21",
        "host_response_rate":1.0,
        "host_acceptance_rate":0.91,
        "host_is_superhost":true,
        "host_has_profile_pic":true,
        "host_identity_verified":false,
        "bathrooms":1,
        "bedrooms":1,
        "beds":1,
        "price":266.0,
        "review_scores_rating":4.5,
        "review_scores_accuracy":4.75,
        "review_scores_cleanliness":4.5,
        "review_scores_checkin":5.0,
        "review_scores_communication":5.0,
        "review_scores_location":5.0,
        "review_scores_value":4.25,
        "license":"",
        "instant_bookable":true,
        "host_response_time":"within an hour",
        "property_type":"Entire rental unit",
        "room_type":"Entire home/apt",
        "amenities":["Kitchen","Wifi","Hot water","TV","Dishes and silverware","Bed linens","Hangers","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Hair dryer","Self check-in","Drying rack for clothing","Blender","Laundromat nearby","Body soap","Luggage dropoff allowed","Shampoo","AC - split type ductless system","Building staff","Outdoor dining area","Ethernet connection","Shared beach access","Outdoor furniture","Patio or balcony","Window guards","Other gas stove","Shared gym in building"]
    },
    ...
    {
        "listing_id":553596330972046819,
        "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
        "host_since":"2020-09-10",
        "host_response_rate":1.0,
        "host_acceptance_rate":0.99,
        "host_is_superhost":true,
        "host_has_profile_pic":true,
        "host_identity_verified":true,
        "bathrooms":1,
        "bedrooms":1,
        "beds":3,
        "price":238.0,
        "review_scores_rating":4.77,
        "review_scores_accuracy":4.85,
        "review_scores_cleanliness":4.8,
        "review_scores_checkin":4.97,
        "review_scores_communication":4.95,
        "review_scores_location":4.96,
        "review_scores_value":4.85,
        "license":"",
        "instant_bookable":true,
        "host_response_time":"within an hour",
        "property_type":"Entire rental unit",
        "room_type":"Entire home/apt",
        "amenities":["Kitchen","Wifi","Hot water","TV","Air conditioning","Dishes and silverware","Bed linens","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Washer","Room-darkening shades","Drying rack for clothing","Stove","Oven","Toaster","Clothing storage","Smoking allowed"]
    }
]
```

#### Ranking search results

```bash
❯ curl -X POST http://localhost:8000/rank \
     -H "Content-Type: application/json" \
     -H "x-user-id: user_9876" \
     -d '[
            {
                "listing_id":1137096251242024922,
                "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
                "host_since":"2023-10-21",
                "host_response_rate":1.0,
                "host_acceptance_rate":0.91,
                "host_is_superhost":true,
                "host_has_profile_pic":true,
                "host_identity_verified":false,
                "bathrooms":1,
                "bedrooms":1,
                "beds":1,
                "price":266.0,
                "review_scores_rating":4.5,
                "review_scores_accuracy":4.75,
                "review_scores_cleanliness":4.5,
                "review_scores_checkin":5.0,
                "review_scores_communication":5.0,
                "review_scores_location":5.0,
                "review_scores_value":4.25,
                "license":"",
                "instant_bookable":true,
                "host_response_time":"within an hour",
                "property_type":"Entire rental unit",
                "room_type":"Entire home/apt",
                "amenities":["Kitchen","Wifi","Hot water","TV","Dishes and silverware","Bed linens","Hangers","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Hair dryer","Self check-in","Drying rack for clothing","Blender","Laundromat nearby","Body soap","Luggage dropoff allowed","Shampoo","AC - split type ductless system","Building staff","Outdoor dining area","Ethernet connection","Shared beach access","Outdoor furniture","Patio or balcony","Window guards","Other gas stove","Shared gym in building"]
            },
            {
                "listing_id":553596330972046819,
                "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
                "host_since":"2020-09-10",
                "host_response_rate":1.0,
                "host_acceptance_rate":0.99,
                "host_is_superhost":true,
                "host_has_profile_pic":true,
                "host_identity_verified":true,
                "bathrooms":1,
                "bedrooms":1,
                "beds":3,
                "price":238.0,
                "review_scores_rating":4.77,
                "review_scores_accuracy":4.85,
                "review_scores_cleanliness":4.8,
                "review_scores_checkin":4.97,
                "review_scores_communication":4.95,
                "review_scores_location":4.96,
                "review_scores_value":4.85,
                "license":"",
                "instant_bookable":true,
                "host_response_time":"within an hour",
                "property_type":"Entire rental unit",
                "room_type":"Entire home/apt",
                "amenities":["Kitchen","Wifi","Hot water","TV","Air conditioning","Dishes and silverware","Bed linens","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Washer","Room-darkening shades","Drying rack for clothing","Stove","Oven","Toaster","Clothing storage","Smoking allowed"]
            }
        ]'
```

```json
# response
{
    "model_used":"model2",
    "ranked_listings":
    [
        {
            "listing_id":553596330972046819,
            "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
            "host_since":"2020-09-10",
            "host_response_rate":1.0,
            "host_acceptance_rate":0.99,
            "host_is_superhost":true,
            "host_has_profile_pic":true,
            "host_identity_verified":true,
            "bathrooms":1,
            "bedrooms":1,
            "beds":3,
            "price":238.0,
            "review_scores_rating":4.77,
            "review_scores_accuracy":4.85,
            "review_scores_cleanliness":4.8,
            "review_scores_checkin":4.97,
            "review_scores_communication":4.95,
            "review_scores_location":4.96,
            "review_scores_value":4.85,
            "license":"",
            "instant_bookable":true,
            "host_response_time":"within an hour",
            "property_type":"Entire rental unit",
            "room_type":"Entire home/apt",
            "amenities":["Kitchen","Wifi","Hot water","TV","Air conditioning","Dishes and silverware","Bed linens","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Washer","Room-darkening shades","Drying rack for clothing","Stove","Oven","Toaster","Clothing storage","Smoking allowed"]
        },
        {
            "listing_id":1137096251242024922,
            "session_id":"295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
            "host_since":"2023-10-21",
            "host_response_rate":1.0,
            "host_acceptance_rate":0.91,
            "host_is_superhost":true,
            "host_has_profile_pic":true,
            "host_identity_verified":false,
            "bathrooms":1,
            "bedrooms":1,
            "beds":1,
            "price":266.0,
            "review_scores_rating":4.5,
            "review_scores_accuracy":4.75,
            "review_scores_cleanliness":4.5,
            "review_scores_checkin":5.0,
            "review_scores_communication":5.0,
            "review_scores_location":5.0,
            "review_scores_value":4.25,
            "license":"",
            "instant_bookable":true,
            "host_response_time":"within an hour",
            "property_type":"Entire rental unit",
            "room_type":"Entire home/apt",
            "amenities":["Kitchen","Wifi","Hot water","TV","Dishes and silverware","Bed linens","Hangers","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Hair dryer","Self check-in","Drying rack for clothing","Blender","Laundromat nearby","Body soap","Luggage dropoff allowed","Shampoo","AC - split type ductless system","Building staff","Outdoor dining area","Ethernet connection","Shared beach access","Outdoor furniture","Patio or balcony","Window guards","Other gas stove","Shared gym in building"]
        }
    ]
}
```

#### Logging user session events

```bash
curl -X POST http://localhost:8000/session-event \
     -H "Content-Type: application/json" \
     -d '[
       {
         "user_uid": "user_9876",
         "session_type": "view_listing",
         "ranking_model": "model2",
         "listing": {
            "listing_id": 553596330972046819,
            "session_id": "295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
            "host_since": "2020-09-10",
            "host_response_rate": 1.0,
            "host_acceptance_rate": 0.99,
            "host_is_superhost": true,
            "host_has_profile_pic": true,
            "host_identity_verified": true,
            "bathrooms": 1,
            "bedrooms": 1,
            "beds": 3,
            "price": 238.0,
            "review_scores_rating": 4.77,
            "review_scores_accuracy": 4.85,
            "review_scores_cleanliness": 4.8,
            "review_scores_checkin": 4.97,
            "review_scores_communication": 4.95,
            "review_scores_location": 4.96,
            "review_scores_value": 4.85,
            "license": "",
            "instant_bookable": true,
            "host_response_time": "within an hour",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "amenities": ["Kitchen","Wifi","Hot water","TV","Air conditioning","Dishes and silverware","Bed linens","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Washer","Room-darkening shades","Drying rack for clothing","Stove","Oven","Toaster","Clothing storage","Smoking allowed"]
        }
       },
       {
         "user_uid": "user_9876",
         "session_type": "view_listing",
         "ranking_model": "model2",
         "listing": {
            "listing_id": 1137096251242024922,
            "session_id": "295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
            "host_since": "2023-10-21",
            "host_response_rate": 1.0,
            "host_acceptance_rate": 0.91,
            "host_is_superhost": true,
            "host_has_profile_pic": true,
            "host_identity_verified": false,
            "bathrooms": 1,
            "bedrooms": 1,
            "beds": 1,
            "price": 266.0,
            "review_scores_rating": 4.5,
            "review_scores_accuracy": 4.75,
            "review_scores_cleanliness": 4.5,
            "review_scores_checkin": 5.0,
            "review_scores_communication": 5.0,
            "review_scores_location": 5.0,
            "review_scores_value": 4.25,
            "license": "",
            "instant_bookable": true,
            "host_response_time": "within an hour",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "amenities": ["Kitchen","Wifi","Hot water","TV","Dishes and silverware","Bed linens","Hangers","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Hair dryer","Self check-in","Drying rack for clothing","Blender","Laundromat nearby","Body soap","Luggage dropoff allowed","Shampoo","AC - split type ductless system","Building staff","Outdoor dining area","Ethernet connection","Shared beach access","Outdoor furniture","Patio or balcony","Window guards","Other gas stove","Shared gym in building"]
        }
       },
       {
         "user_uid": "user_9876",
         "session_type": "book_listing",
         "ranking_model": "model2",
         "listing": {
            "listing_id": 553596330972046819,
            "session_id": "295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649",
            "host_since": "2020-09-10",
            "host_response_rate": 1.0,
            "host_acceptance_rate": 0.99,
            "host_is_superhost": true,
            "host_has_profile_pic": true,
            "host_identity_verified": true,
            "bathrooms": 1,
            "bedrooms": 1,
            "beds": 3,
            "price": 238.0,
            "review_scores_rating": 4.77,
            "review_scores_accuracy": 4.85,
            "review_scores_cleanliness": 4.8,
            "review_scores_checkin": 4.97,
            "review_scores_communication": 4.95,
            "review_scores_location": 4.96,
            "review_scores_value": 4.85,
            "license": "",
            "instant_bookable": true,
            "host_response_time": "within an hour",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "amenities": ["Kitchen","Wifi","Hot water","TV","Air conditioning","Dishes and silverware","Bed linens","Cooking basics","Microwave","Essentials","Elevator","Refrigerator","Dedicated workspace","Washer","Room-darkening shades","Drying rack for clothing","Stove","Oven","Toaster","Clothing storage","Smoking allowed"]
        }
       }
     ]'
```

```json
# response
{"status":"ok"}
```

**`results.csv` file:**
```csv
user_uid,session_type,ranking_model,listing_id,session_id,host_since,host_response_rate,host_acceptance_rate,host_is_superhost,host_has_profile_pic,host_identity_verified,bathrooms,bedrooms,beds,price,review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,review_scores_value,license,instant_bookable,host_response_time,property_type,room_type,amenities
user_9876,view_listing,model2,553596330972046819,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2020-09-10,1.0,0.99,True,True,True,1,1,3,238.0,4.77,4.85,4.8,4.97,4.95,4.96,4.85,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Air conditioning', 'Dishes and silverware', 'Bed linens', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Washer', 'Room-darkening shades', 'Drying rack for clothing', 'Stove', 'Oven', 'Toaster', 'Clothing storage', 'Smoking allowed']"
user_9876,view_listing,model2,1137096251242024922,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2023-10-21,1.0,0.91,True,True,False,1,1,1,266.0,4.5,4.75,4.5,5.0,5.0,5.0,4.25,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Dishes and silverware', 'Bed linens', 'Hangers', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Hair dryer', 'Self check-in', 'Drying rack for clothing', 'Blender', 'Laundromat nearby', 'Body soap', 'Luggage dropoff allowed', 'Shampoo', 'AC - split type ductless system', 'Building staff', 'Outdoor dining area', 'Ethernet connection', 'Shared beach access', 'Outdoor furniture', 'Patio or balcony', 'Window guards', 'Other gas stove', 'Shared gym in building']"
user_9876,book_listing,model2,553596330972046819,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2020-09-10,1.0,0.99,True,True,True,1,1,3,238.0,4.77,4.85,4.8,4.97,4.95,4.96,4.85,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Air conditioning', 'Dishes and silverware', 'Bed linens', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Washer', 'Room-darkening shades', 'Drying rack for clothing', 'Stove', 'Oven', 'Toaster', 'Clothing storage', 'Smoking allowed']"
```
