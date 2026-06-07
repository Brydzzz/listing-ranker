# Listing Ranker

## Struktura repozytorium

```sh
.
├── ab_analysis.ipynb # skrypt do analizy wyników eksperymentu
├── app # pliki mikroserwisu
│   ├── example_flow.py # przykładowe wykorzystanie mikroserwisu
│   ├── __init__.py
│   ├── listings_router.py # APIRouter do listings
│   ├── model_selector.py # implementacja consistent hashing do przydzielania modelu
│   ├── models.py # implementacja inferencji dwoch modeli bazowego i docelowego
│   ├── rank_router.py # APIRouter do uzyskania porankingownej listy ofert
│   ├── schemas.py # schemat Listing
│   └── sessions_router.py # APIRouter do logowania sesji uzytkownika
├── assets # zdjęcie do dokumentacji
│   ├── ...
├── column_embedding.py # skrypt do tworzenia zagnieżdżeń name i description
├── combine_data.py # skrypt łączący i przerabiający dane
├── extract_ab_data.py # wydzielenie danych zwróconych przez combine_data na test A/B
├── IUM - Etap 2 - Raport z procesu budowy modelu.md # raport z procesu budowy modelu z porownaniem wynikow
├── IUM - Etap 2 - Raport z procesu budowy modelu.pdf # raport w wersji pdf
├── main.py # FastAPI app, uruchomienie `uvicorn main:app`
├── pyproject.toml
├── README.md
├── README.pdf # ten plik w wersji pdf
├── run_lgbm_experiments_amenitites_on.py # skrypt uruchamiający różne konfigurację z amenitites
├── run_lgbm_experiments_params.py # skrypt uruchamiający różne konfigurację z parametrów lgbm
├── run_lgbm_experiments.py # skrypt uruchamiający różne konfigurację atrybutów
├── run_log_amen.txt # wyniki run_lgbm_experiments_amenitites_on.py
├── run_log_params.txt # wyniki run_lgbm_experiments_params.py
├── run_log.txt # wyniki run_lgbm_experiments.py
├── single_lgbm_run.py # pojedynczy uruchomienie treningu modelu
├── simulate_ab.py # skrypt symulający większą ilość danych z a/b
└── uv.lock
```

## Pliki tworzone przez skrypty i potrzebne do skryptów

```sh
.
├── combined_ab.parquet # wynik extract_ab_data.py, wykorzystywane w mikroserwisie
├── combined_no_ab.parquet # wynik extract_ab_data.py, wykorzystywane w single_lgbm_run.py
├── combined.parquet # wynik combine_data.py
├── embeddings.npz # zwracany przez column_embedding.py, wykorzystywane w single_lgbm_run.py
├── eval_dataset.bin # tworzony w trakcie single_lgbm_run, umozliwia ponowne uruchomienie treningu bez potrzeby ponownej budowy lgb.Dataset
├── lgbm_model_with_feature_names.txt # zapisany model
├── listing_ids.npy # zwracany przez column_embedding.py, wykorzystywane w single_lgbm_run.py
├── metadata_cache.pkl # tworzony w trakcie single_lgbm_run, umozliwia ponowne uruchomienie treningu bez potrzeby ponownej budowy lgb.Dataset
├── results.csv # wyniki mikroserwisu
├── listings.csv # potrzebne do utworzenia combined.parquet i zagnieżdżeń
├── sessions.csv # potrzebne do utworzenia combined.parquet
├── test_data_cache.parquet # tworzony w trakcie single_lgbm_run, umozliwia ponowne uruchomienie treningu bez potrzeby ponownej budowy lgb.Dataset
└── train_dataset.bin # tworzony w trakcie single_lgbm_run, umozliwia ponowne uruchomienie treningu bez potrzeby ponownej budowy lgb.Dataset
```

## O mikroserwisie

- Endpoint /listings symuluje listingi zwrócone przez silnik wyszukujący, w naszym mikroserwisie zwraca listingi które były w tej samej sesji.

- endpoint /session-event służy do logowania wyników testu a/b do późniejszej analizy

- użytkownikowi przypisywany jest konkretny model za pomocą funkcji hashującej

- Plik `example_flow.py` pokazuje przykładowe korzystanie z mikroserwisu

- Plik `simulate_ab` generuje syntetyczne wyniki testu ab do analizy przez `ab_analysis.ipynb`

- plik `ab_analysis.ipynb` pozwala na zweryfikowanie kryterium biznesowego poprzez pokazanie średniej długości sesji

### Przykład użycia - `curl` i analiza w jupyter notebooku

#### Start mikroserwisu

Najpierw należy uruchomić mikroserwis: `uvicorn main:app`
```bash
❯ uvicorn main:app
INFO:     Started server process [7785]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### Uzyskanie listy ofert symulującej wyniki wyszukiwania:

```bash
❯ curl -s http://localhost:8000/listings
```

```json
# odpowiedź
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

#### Rankowanie wyników wyszukiwania

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
# odpowiedź
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

#### Logowanie eventów sesji użytkownika

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
# odpowiedź
{"status":"ok"}
```

**Plik results.csv:**
```csv
user_uid,session_type,ranking_model,listing_id,session_id,host_since,host_response_rate,host_acceptance_rate,host_is_superhost,host_has_profile_pic,host_identity_verified,bathrooms,bedrooms,beds,price,review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,review_scores_value,license,instant_bookable,host_response_time,property_type,room_type,amenities
user_9876,view_listing,model2,553596330972046819,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2020-09-10,1.0,0.99,True,True,True,1,1,3,238.0,4.77,4.85,4.8,4.97,4.95,4.96,4.85,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Air conditioning', 'Dishes and silverware', 'Bed linens', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Washer', 'Room-darkening shades', 'Drying rack for clothing', 'Stove', 'Oven', 'Toaster', 'Clothing storage', 'Smoking allowed']"
user_9876,view_listing,model2,1137096251242024922,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2023-10-21,1.0,0.91,True,True,False,1,1,1,266.0,4.5,4.75,4.5,5.0,5.0,5.0,4.25,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Dishes and silverware', 'Bed linens', 'Hangers', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Hair dryer', 'Self check-in', 'Drying rack for clothing', 'Blender', 'Laundromat nearby', 'Body soap', 'Luggage dropoff allowed', 'Shampoo', 'AC - split type ductless system', 'Building staff', 'Outdoor dining area', 'Ethernet connection', 'Shared beach access', 'Outdoor furniture', 'Patio or balcony', 'Window guards', 'Other gas stove', 'Shared gym in building']"
user_9876,book_listing,model2,553596330972046819,295857ea36653d4efac9c70ad7186cfff4c0237282f1aa6478584441530fa649,2020-09-10,1.0,0.99,True,True,True,1,1,3,238.0,4.77,4.85,4.8,4.97,4.95,4.96,4.85,,True,within an hour,Entire rental unit,Entire home/apt,"['Kitchen', 'Wifi', 'Hot water', 'TV', 'Air conditioning', 'Dishes and silverware', 'Bed linens', 'Cooking basics', 'Microwave', 'Essentials', 'Elevator', 'Refrigerator', 'Dedicated workspace', 'Washer', 'Room-darkening shades', 'Drying rack for clothing', 'Stove', 'Oven', 'Toaster', 'Clothing storage', 'Smoking allowed']"
```

**Analiza `results.csv` w `ab_analysis.ipynb`**

![wyniki ipynb curl example](assets/curl_results.png)

### Wyniki `ab_analysis.ipynb` dla `example_flow.py`

![wyniki ipynb exmaple_flow](assets/exmaple_flow_results.png)

### Wyniki `ab_analysis.ipyb` dla `simulate_ab.py`

![wyniki ipynb simualte ab](assets/simulate_ab_results.png)
