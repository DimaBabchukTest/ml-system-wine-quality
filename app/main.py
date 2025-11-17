#uv run uvicorn app.main:app --reload

import pickle
import pandas as pd
from fastapi import FastAPI, Request

model_file = './model_artifact/wine_rate_v1.bin'

with open(model_file, 'rb') as f_in:
    pipline = pickle.load(f_in)

app = FastAPI()

@app.post('/predict')
async def predict(request: Request):
    wine_details = await request.json()
    print(wine_details)
    df = pd.DataFrame([wine_details])
    final_features_without_target = ['volatile_acidity', 'citric_acid', 'free_sulfur_dioxide',
    'total_sulfur_dioxide','density', 'ph', 'sulphates', 'alcohol']

    wine_record = df[final_features_without_target]
    print(wine_record)
    y_pred = pipline.predict_proba(wine_record)[0, 1]
    treshold = 0.5 # in order to get good wine for sure 
    result = {
        'Good?': (round(float(y_pred),3) > 0.5),
        'prediction_probability':  round(float(y_pred),3),        
        'request': wine_details
    }

    return result

