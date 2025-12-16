import os
import sys

# ================= LOAD ENV FIRST =================
from dotenv import load_dotenv
load_dotenv()

# ================= BASIC SETUP ====================
import certifi
ca = certifi.where()

mongo_db_url = os.getenv("MONGO_DB_URL")
if not mongo_db_url:
    raise ValueError("MONGO_DB_URL not found. Check .env file")

print("MongoDB URL loaded successfully")

# ================= DATABASE =======================
import pymongo
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ================= APP IMPORTS ====================
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ================= FASTAPI ========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

# ================= ROUTES =========================
@app.get("/", tags=["root"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        predictions = network_model.predict(df)
        df["prediction"] = predictions

        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html},
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ================= RUN ============================
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
