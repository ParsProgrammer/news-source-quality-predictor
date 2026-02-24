import json
import os
import time
import uuid
import boto3
import pandas as pd

# ----- Config -----
REGION = os.environ.get("AWS_REGION", "eu-north-1")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "nsq-endpoint-20260208-120029")
CUES_CSV_PATH = os.environ.get("CUES_CSV_PATH", "E:/NSQ_predictor/back_end/data/cues_US.csv")

# Your feature list (must match inference.py)
partisan_all_cues = [
    'broadcasters_inequality_all','broadcasters_inequality_left','broadcasters_inequality_right','broadcasters_inequality_minima',
    'article_inequality_all','article_inequality_left','article_inequality_right','article_inequality_minima',
    'centrality_degree_all','centrality_degree_left','centrality_degree_right','centrality_degree_minima',
    'centrality_eigen-vector_all','centrality_eigen-vector_left','centrality_eigen-vector_right','centrality_eigen-vector_minima',
    'centrality_page-rank_all','centrality_page-rank_left','centrality_page-rank_right','centrality_page-rank_minima',
    'centrality_harmonic_all','centrality_harmonic_left','centrality_harmonic_right','centrality_harmonic_minima',
    'centrality_closeness_all','centrality_closeness_left','centrality_closeness_right','centrality_closeness_minima',
    'centrality_betweenness_all','centrality_betweenness_left','centrality_betweenness_right','centrality_betweenness_minima',
    'broadcasts_num-log_all','broadcasts_num-log_left','broadcasts_num-log_right',
    'broadcasters_num-log_all','broadcasters_num-log_left','broadcasters_num-log_right',
    'broadcasts_prop-left','broadcasters_prop-left'
]
FEATURES = ["diversity_qn"] + partisan_all_cues

_smr = None
_df = None

def _get_smr():
    global _smr
    if _smr is None:
        _smr = boto3.client("sagemaker-runtime", region_name=REGION)
    return _smr

def normalize_domain(s: str) -> str:
    d = s.strip().lower()
    d = d.replace("http://", "").replace("https://", "")
    d = d.split("/")[0]
    return d

def load_df():
    """Load CSV once per process."""
    global _df
    if _df is None:
        if not os.path.exists(CUES_CSV_PATH):
            raise RuntimeError(f"CUES_CSV_PATH not found: {CUES_CSV_PATH}")
        _df = pd.read_csv(CUES_CSV_PATH)
    return _df

def get_features_for_domain(domain: str) -> dict:
    """
    Looks up a domain row in your CSV and returns {feature: value,...}.
    IMPORTANT: You must set DOMAIN_COLUMN below to the actual column name in cues_US.csv.
    """
    df = load_df()

    DOMAIN_COLUMN = os.environ.get("DOMAIN_COLUMN", "domain")  # change if your CSV uses a different name

    if DOMAIN_COLUMN not in df.columns:
        raise RuntimeError(
            f"DOMAIN_COLUMN='{DOMAIN_COLUMN}' not found in CSV columns. "
            f"Available columns include: {list(df.columns)[:15]}..."
        )

    d = normalize_domain(domain)

    # Exact match
    row = df.loc[df[DOMAIN_COLUMN].astype(str).str.lower() == d]

    # Optional: try stripping leading www.
    if row.empty and d.startswith("www."):
        d2 = d[4:]
        row = df.loc[df[DOMAIN_COLUMN].astype(str).str.lower() == d2]
        d = d2

    if row.empty:
        return {"error": "domain_not_found", "domain": d}

    # Take first match if duplicates exist
    r0 = row.iloc[0]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required feature columns: {missing}")

    # Build feature dict and coerce to float
    feats = {}
    for f in FEATURES:
        v = r0[f]
        try:
            feats[f] = float(v)
        except Exception:
            raise RuntimeError(f"Non-numeric value for feature '{f}' on domain '{d}': {v!r}")

    return {"domain": d, "features": feats}

def invoke_sagemaker(features: dict) -> float:
    payload = {"instances": [features]}
    smr = _get_smr()

    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )

    out = resp["Body"].read().decode("utf-8")
    data = json.loads(out) if out else {}
    preds = data.get("predictions", data)

    # handle {"predictions":[...]} or plain list
    if isinstance(preds, list):
        return float(preds[0])
    return float(preds)

def predict_domain(domain: str) -> dict:
    t0 = time.time()
    req_id = str(uuid.uuid4())

    got = get_features_for_domain(domain)
    if "error" in got:
        return {"error": got["error"], "domain": got["domain"], "request_id": req_id}

    pred = invoke_sagemaker(got["features"])

    return {
        "domain": got["domain"],
        "prediction": pred,
        "request_id": req_id,
        "latency_ms": int((time.time() - t0) * 1000),
    }