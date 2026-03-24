import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import output_recommended_recipes, recommend

BASE_DIR = Path(__file__).resolve().parent
for _candidate in [BASE_DIR / "Data" / "dataset.csv.gz", BASE_DIR / "Data" / "dataset.csv"]:
    if _candidate.is_file():
        dataset = pd.read_csv(_candidate, compression="gzip" if _candidate.suffix == ".gz" else None)
        break
else:
    raise FileNotFoundError("Data/dataset.csv.gz not found")

app = FastAPI(title="HealthEats API", docs_url="/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input: List[float] = Field(..., min_length=9, max_length=9)
    ingredients: List[str] = []
    ingredients_to_avoid: List[str] = []
    ingredients_to_avoid_txt: Optional[List[str]] = None
    params: Optional[Params] = None


class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]


class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
@app.get("/health")
def health():
    return {"health_check": "OK"}


@app.post("/predict/", response_model=PredictionOut)
def predict(prediction_input: PredictionIn):
    p = prediction_input.params.model_dump() if prediction_input.params else {"n_neighbors": 5, "return_distance": False}
    avoid = [x for x in (prediction_input.ingredients_to_avoid_txt or prediction_input.ingredients_to_avoid) if x and str(x).strip()]
    df = recommend(dataset, list(prediction_input.nutrition_input), prediction_input.ingredients, avoid, p)
    output = output_recommended_recipes(df)
    return PredictionOut(output=output)
