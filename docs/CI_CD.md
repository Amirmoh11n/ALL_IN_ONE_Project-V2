# CI / CD

## CI (continuous integration)

On every push and pull request, GitHub Actions (`.github/workflows/ci.yml`) runs:

1. **Lint** — `ruff check src tests webapplication`
2. **Tests** — `pytest -q` (unit + API, no dataset download)
3. **Docker build** — serving image from `docker/Dockerfile` (lightweight web deps)

CI answers: *does this commit still work?* It does not train the model and does
not push to a hospital/cloud registry unless you add that later.

Local equivalent:

```bash
PYTHONPATH=. pytest -q
ruff check src tests webapplication
docker build -f docker/Dockerfile -t neuramri-web:local .
```

## CD (continuous delivery)

V2 CD is **build the serving image after CI is green**. There is no automatic
production deploy (localhost / Docker-first).

Typical delivery path:

1. Train + evaluate + export ONNX on a GPU machine
2. Place `artifacts/exports/brain_tumor_efficientnet_b4.onnx` in the repo *or*
   mount it at runtime (`MODEL_PATH`)
3. `docker compose` / `docker build` from `docker/`
4. Health check: `GET /api/health` then `POST /api/predict`

Optional later CD (not enabled by default):

- Push the image to GHCR/ECR only on tags (`v2.*`)
- Deploy to a staging VM after a successful tag build
- Never auto-deploy unreviewed medical-research models to a public URL
