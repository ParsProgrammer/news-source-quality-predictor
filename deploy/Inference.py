import json
import os
import time
import logging
import joblib
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- feature lists (kept from your code) ----
partisan_all_cues = [
    'broadcasters_inequality_all','broadcasters_inequality_left','broadcasters_inequality_right',
    'broadcasters_inequality_minima','article_inequality_all','article_inequality_left',
    'article_inequality_right','article_inequality_minima','centrality_degree_all',
    'centrality_degree_left','centrality_degree_right','centrality_degree_minima',
    'centrality_eigen-vector_all','centrality_eigen-vector_left','centrality_eigen-vector_right',
    'centrality_eigen-vector_minima','centrality_page-rank_all','centrality_page-rank_left',
    'centrality_page-rank_right','centrality_page-rank_minima','centrality_harmonic_all',
    'centrality_harmonic_left','centrality_harmonic_right','centrality_harmonic_minima',
    'centrality_closeness_all','centrality_closeness_left','centrality_closeness_right',
    'centrality_closeness_minima','centrality_betweenness_all','centrality_betweenness_left',
    'centrality_betweenness_right','centrality_betweenness_minima','broadcasts_num-log_all',
    'broadcasts_num-log_left','broadcasts_num-log_right','broadcasters_num-log_all',
    'broadcasters_num-log_left','broadcasters_num-log_right','broadcasts_prop-left',
    'broadcasters_prop-left'
]

FEATURES = ["diversity_qn"] + partisan_all_cues
MODEL_FILENAME = "elastic_net.joblib"


# ---- SageMaker required hooks ----
def model_fn(model_dir: str):
    """Load the model from the SageMaker model directory (/opt/ml/model)."""
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def input_fn(request_body: str, request_content_type: str):
    """Parse JSON request into a pandas DataFrame with required columns."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    # Expect {"instances": [ {col: val, ...}, ... ]}
    if "instances" not in payload:
        raise ValueError("Request JSON must contain top-level key 'instances'")

    instances = payload["instances"]
    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError("'instances' must be a non-empty list")

    df = pd.DataFrame(instances)

    # Validate required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Ensure correct ordering and numeric
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    return X


def predict_fn(X: pd.DataFrame, model):
    """Run prediction and emit simple structured logs (CloudWatch)."""
    start = time.time()
    preds = model.predict(X)
    latency_ms = int((time.time() - start) * 1000)

    logger.info(json.dumps({
        "event": "inference",
        "latency_ms": latency_ms,
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "model_file": MODEL_FILENAME
    }))

    return preds


def output_fn(preds, accept: str):
    """Return JSON predictions."""
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    # Convert numpy -> plain Python
    preds_list = [float(x) for x in preds]
    return json.dumps({"predictions": preds_list}), "application/json"
