from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from invoke_service import predict_domain

app = FastAPI()

# Dev CORS: allow Next.js local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictReq(BaseModel):
    domain: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictReq):
    res = predict_domain(req.domain)

    if res.get("error") == "domain_not_found":
        raise HTTPException(status_code=404, detail="domain_not_found")

    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])

    return res