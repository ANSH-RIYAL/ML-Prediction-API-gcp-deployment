# ML Prediction API - GCP Cloud Run Deployment

Lifespan prediction and disease risk model API. Deployed as a Cloud Run service.

## Endpoints

- `GET /health` — Health check
- `GET /test/` — Health check (alternate)
- `POST /prediction_model_test/` — Predict lifespan and disease risk ratios
- `POST /optimization_model_test/` — Get optimal variable values

## Prediction Request

```json
POST /prediction_model_test/
{
  "age": 30,
  "gender": 0,
  "fruits_and_veggies": 6,
  "dietary_fiber": 25,
  "olive_oil": 30
}
```

## Prediction Response

```json
{
  "all_cause_mortality_predicted_lifespan": 96.56,
  "cancer_predicted_rr": 0.61,
  "cardio_vascular_disease_predicted_rr": 0.25,
  "depression_predicted_rr": 1.0,
  "diabetes_predicted_rr": 0.46,
  "stroke_predicted_rr": 0.12
}
```

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

Auto-deploys to Cloud Run on push to `main` via GitHub Actions.

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_SA_KEY` | GCP service account key JSON |
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region (e.g., `us-central1`) |
