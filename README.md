
# Wine Quality ML Model (End-to-end data science workflow + FastAPI + Docker + uv)

This repository provides:

- A **machine learning model** for predicting wine quality  
- A **FastAPI prediction service**  
- A reproducible environment using **uv**  
- A **Dockerized API** ready for deployment  
- A complete end-to-end data science workflow (see **[a relative link](jupyter_notebook/Wine_Rate.ipynb))**

---

## 1. Dataset Description

This project uses the **Wine Quality Dataset** from the  **UCI Machine Learning Repository**  
[a link](https://archive.ics.uci.edu/dataset/186/wine+quality)
(copyright belongs to the original authors; see `wine_quality_data/winequality.names`).

The dataset contains physicochemical measurements for Portuguese red and white Vinho Verde wines, collected from real production environments. Each sample includes laboratory‑measured attributes (acidity, sugar, sulfur dioxide, pH, alcohol, etc.) and a sensory quality score assigned by professional tasters.

### Binary Classification Target

```
good_wine = 1  if  quality >= 5  good_wine = 0  otherwise
```

This reflects real industrial processes where the question is not “exact score” but whether a wine meets minimum commercial quality.

### Real‑World Use Cases

- **Wineries / production lines**  
  Automatically score wine batches and adjust fermentation parameters, sulfur dioxide, or sugar levels.

- **Automated quality control**  
  Predict whether a wine batch meets acceptable standards before sending to expert tasters.

- **Pricing support**  
  Predicted quality probability can inform price segmentation.

### Dataset Files in Repository

```
wine_quality_data/
│── winequality-red.csv
│── winequality-white.csv
└── winequality.names
```

More preprocessing details are in:  
`jupyter_notebook/Wine_Rate.ipynb`

---

## 2. EDA Summary

A full EDA was conducted including distribution analysis, correlation patterns, dimensionality reduction, and outlier evaluation.

### Key Insights

- **1,179 duplicate rows removed**  
- **No missing values** found  
- Dataset is **moderately imbalanced**  
- Many features show **right-skewed** distributions (e.g., residual sugar, sulphates)  
- Alcohol shows a **strong shift upward** for good wines  
- **Moderate correlations** were found:  
  - free_sulfur_dioxide ↔ total_sulfur_dioxide  
  - density ↔ alcohol  
  - density ↔ residual_sugar  
- PCA, LDA, t‑SNE show the classes are **not linearly separable**  
- Outliers were **kept** because they are chemically reasonable and tree models handle them well

Full plots and diagnostics are in:  
📘 `jupyter_notebook/Wine_Rate.ipynb`

---

## 3. Model Training Summary

Three main model families were trained & compared using identical **stratified** splits:

1. **Logistic Regression**  
   - Standardized features  
   - Tuned over C  
   - Baseline model

2. **Random Forest**  
   - Tuned over n_estimators, max_depth, min_samples_leaf, min_samples_split  
   - Best CV AUC: **0.8316 ± 0.0063**

3. **XGBoost**  
   - Tuned via randomized search  
   - Best CV AUC: **0.8305 ± 0.0063**


`RANDOM_SEED = 42` was used for full reproducibility

### Model Selection Strategy

- **Primary metric:** ROC AUC (cross‑validated mean ± std)  
- **Secondary metric:** F1 Score  
- All models evaluated on identical folds

### Winner
The **Random Forest** performed best considering AUC, stability, and simplicity:

```
CV AUC  ≈ 0.8316 ± 0.0063  
Val AUC ≈ 0.8337  
```

Best parameters:

```
max_depth = 30
n_estimators = 120
min_samples_leaf = 5
min_samples_split = 10
```

### Feature Reduction

Based on AUC drop tests and subset evaluation:

Least useful features:  
`chlorides`, `fixed_acidity`, `residual_sugar`

Final compact feature set (8 features):

```
['volatile_acidity', 'citric_acid', 'free_sulfur_dioxide',
 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']
```

Performance remained nearly identical -> simpler, faster, and cleaner model.

---

## 4. Calibration & Threshold Selection

The final Random Forest model was:

### Calibrated using Isotonic Regression  
(`CalibratedClassifierCV(method="sigmoid", cv="prefit")`)

###  Threshold Optimization

Thresholds between 0 and 1 were evaluated on the threshould data set.  
The F1‑optimal threshold:

```
≈ 0.37
```
F1 (t = 0.37) = 0.8284182305630027

### Test Performance on calibrated model

- **ROC AUC ≈ 0.8218**  
- **F1 ≈ 0.8192** (threshold 0.37)  
- **F1 ≈ 0.8228** (threshold 0.50)
- **Brier Score  = 0.1640** 

(see [a relative link](jupyter_notebook/Wine_Rate.ipynb))

> **Note:**
For production use, a more conservative threshold (e.g. ≥ 0.50) is recommended to prioritize precision over recall.

# (Training + FastAPI + Docker + uv)

- A **FastAPI service** to expose predictions
- A reproducible environment using **uv**
- A **Dockerized API service** ready for deployment
- A full **training pipeline** to retrain the model

------------------------------------------------------------------------

## 1. Project Structure

```
WineQuality/
│
├── app/
│   └── main.py                 # API serving script
│
├── train_eval_model/
│   └── train_eval_model.py     # Training script
│
├── model_artifact/
│   └── wine_rate_v1.bin        # Stored model after training
│
├── requests/
│   └── request_local.py        # Prediction example
│
├── jupyter_notebook/
│   └── Wine_Rate.ipynb         # Full analysis and training notebook
│
├── wine_quality_data/          # Raw dataset
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
│
├── pyproject.toml
├── uv.lock
└── Dockerfile
```
> **Note:**  
 This project follows a modular ML layout: 
- `train_eval_model.py` performs model training
- `main.py` serves predictions via FastAPI
- `request_localy.py` provides an example prediction request  

This mirrors the common ML pattern of **train.py → predict.py → serve.py** used in production systems.


------------------------------------------------------------------------

## 2. Clone the Project

``` bash
git clone https://github.com/DimaBabchukTest/ml-system-wine-quality.git
cd WineQuality
```

------------------------------------------------------------------------

## 3. Install uv

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Restart terminal if needed.

------------------------------------------------------------------------

## 4. Recreate Environment Using uv

``` bash
uv sync --locked
uv lock --check
```

This creates a `.venv` and installs all pinned dependencies.

------------------------------------------------------------------------

## 5. Train the Model

To retrain the wine ML model:

``` bash
uv run python -m train_eval_model.train_eval_model
```

The trained model will be saved to:

    model_artifact/wine_rate_v1.bin

------------------------------------------------------------------------

## 6. Run FastAPI Locally

Start the API:

``` bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

-   API root: http://127.0.0.1:8000\
-   Swagger UI: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## 7. Test Prediction Locally

Run request script:

``` bash
uv run python requests/request_localy.py
```

This sends a request to the running FastAPI app and prints the
prediction.

------------------------------------------------------------------------

## 8. Docker Instructions

### Install Docker (if needed)

https://www.docker.com/products/docker-desktop/

Verify:

``` bash
docker --version
```

------------------------------------------------------------------------

### Build Docker Image

``` bash
docker build -t wine_model_docker .
```

------------------------------------------------------------------------

### Run Docker Container

``` bash
docker run -p 127.0.0.1:8000:8000 --name wine_model_container wine_model_docker
```

API is now available at:

-   http://127.0.0.1:8000\
-   http://127.0.0.1:8000/docs

------------------------------------------------------------------------

### Test Prediction via Docker

``` bash
uv run python requests/request_localy.py
```

------------------------------------------------------------------------

### Stop and Remove Container + Image

``` bash
docker stop wine_model_container
docker rm wine_model_container
docker rmi wine_model_docker
```

------------------------------------------------------------------------

## 9. Summary Commands

### Local (uv)

``` bash
uv sync --locked
uv run python -m train_eval_model.train_eval_model
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run python requests/request_localy.py
```

### Docker

``` bash
docker build -t wine_model_docker .
docker run -p 127.0.0.1:8000:8000 --name wine_model_container wine_model_docker

> **IMPORTANT:**
Do not stop the FastAPI server.  
Open a **second terminal**, navigate to the project folder again, and run:

uv run python requests/request_localy.py

docker stop wine_model_container
docker rm wine_model_container
docker rmi wine_model_docker
```

------------------------------------------------------------------------

## 10. Deployment

This project can be deployed easily to:

-   Render (recommended free tier) (Example of deployed model and result of prediction please see  `presentation` folder)
-   Koyeb (free instance available)
-   Fly.io (low cost, not fully free)

Dockerfile is already configured for deployment.

