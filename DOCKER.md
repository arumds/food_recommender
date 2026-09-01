## Build the image

From the project root (where `Dockerfile` lives):

```bash
docker build -t wolt-reco-api .
```

This installs only `requirements-serve.txt` (6 packages: fastapi, uvicorn, numpy, pandas,
faiss-cpu, lightgbm) and copies in only the 4 source files `serve.py` actually needs
(`app.py`, `data_loader.py`, `ranking_fe.py`, `balance_relevance_constraints.py`). It does **not** include
`data/` or `models/` — those get mounted in at run time (see below).

## Run the container

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  wolt-reco-api
```
## Test it

From another terminal (not the one running the container):

```bash
curl "http://0.0.0.0:8000/health"
curl "http://0.0.0.0:8000/recommend?user_id=0&hour=19&k=10"
```

Or open `http://0.0.0.0:8000/docs` in a browser for FastAPI's interactive API explorer.

## Stopping the container

`Ctrl+C` in the terminal running it, or from another terminal:

```bash
docker ps                  # find the container ID or name
docker stop <container_id_or_name>
```
