from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Load the model at startup
with open("stock-price.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input data model
class StockData(BaseModel):
    day: int
    month: int
    year: int
    Open: float
    High: float
    Low: float
    Adj_Close: float
    Volume: int

    @validator("day", "month", "year")
    def check_date(cls, v, values, field):
        if "day" in values and "month" in values and "year" in values:
            try:
                # Check if the date is valid
                datetime(year=values["year"], month=values["month"], day=values["day"])
            except ValueError:
                raise ValueError(f"Invalid date: {values['day']}-{values['month']}-{values['year']}")
        return v

# Add CORS middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_next_5_days(data: StockData):
    # Convert the input data to a DataFrame and ensure proper types
    input_data = pd.DataFrame([data.dict()])
    input_data = input_data.astype({
        'day': int,
        'month': int,
        'year': int,
        'Open': float,
        'High': float,
        'Low': float,
        'Adj_Close': float,
        'Volume': int
    })

    # Generate the next 5 days of data
    predictions = []
    for _ in range(5):
        # Predict the Close price
        predicted_close = model.predict(input_data)[0]

        # Append the prediction to the results
        predictions.append({
            "day": int(input_data.loc[0, "day"]),
            "month": int(input_data.loc[0, "month"]),
            "year": int(input_data.loc[0, "year"]),
            "predicted_close": float(predicted_close)
        })

        # Prepare the input data for the next day
        next_date = datetime(int(input_data.loc[0, "year"]), int(input_data.loc[0, "month"]), int(input_data.loc[0, "day"])) + timedelta(days=1)
        input_data.loc[0, "day"] = next_date.day
        input_data.loc[0, "month"] = next_date.month
        input_data.loc[0, "year"] = next_date.year
        input_data.loc[0, "Open"] = predicted_close
        input_data.loc[0, "High"] = predicted_close  # Assuming High is the same as predicted Close
        input_data.loc[0, "Low"] = predicted_close  # Assuming Low is the same as predicted Close
        input_data.loc[0, "Adj_Close"] = predicted_close  # Assuming Adj_Close is the same as predicted Close

    return predictions

if __name__ == "__main__":
    uvicorn.run(app)
