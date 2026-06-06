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
│   ├── 2026-06-04-15-38-45-image.png
│   ├── 2026-06-04-15-39-46-image.png
│   ├── 2026-06-04-15-48-20-image.png
│   └── 4b80998f-f7a5-44fa-a463-d7ba8846c9f0.jpeg
├── column_embedding.py # skrypt do tworzenia zagnieżdżeń name i description
├── combine_data.py # skrypt łączący i przerabiający dane
├── data_analysis.ipynb # NIE WYSLYAC W ZIP
├── data_analysis_final.ipynb # NIE WYSLYAC W ZIP
├── extract_ab_data.py # wydzielenie danych zwróconych przez combine_data na test A/B
├── IUM - Etap 2 - Raport z procesu budowy modelu.md # raport z procesu budowy modelu z porownaniem wynikow
├── main.py # FastAPI app
├── pyproject.toml
├── README.md
├── run_lgbm_experiments_amenitites_on.py # skrypt uruchamiający różne konfigurację z amenitites
├── run_lgbm_experiments_params.py # skrypt uruchamiający różne konfigurację z parametrów lgbm
├── run_lgbm_experiments.py # skrypt uruchamiający różne konfigurację atrybutów
├── run_log_amen.txt # wyniki run_lgbm_experiments_amenitites_on.py
├── run_log_params.txt # wyniki run_lgbm_experiments_params.py
├── run_log.txt # wyniki run_lgbm_experiments.py
├── single_lgbm_run.py # pojedynczy uruchomienie treningu modelu
├── simple_model_only_tf.py # NIE WYSYLAC W ZIP
├── simulate_ab.py # NIE WIEM CO Z TYM CZY TO WYSYLAM CZY NIE
├── test_n_rows_data_distribution.py # NIE WYSYLAC W ZIP
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

- endpoint /session-event sluzy do logowania wyników testu a/b do późniejszej analizy

- użytkownikowi przypisywany jest konkrenty model za pomocą funkcji hashującej

- Plik `example_flow.py` pokazuje przykładowe korzystane z mikroserwisu.

- Plik `simulate_ab` generuje syntetyczne wyniki testu ab do analizy przez `ab_analysis.ipynb`

- plik `ab_analysis.ipynb` pozwala na zweryfikowanie kryterium biznesowego poprzez pokazanie średniej dlugości sesji