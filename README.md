# tip-prediction-nyc🗽
## Objetivo // Objective

[ **ES** ]
- Desarrollar un proyecto de ciencia de datos siguiendo un flujo de trabajo estructurado basado en la metodología Cookiecutter Data Science.
- Construir un modelo de machine learning para predecir viajes en taxi en la ciudad de Nueva York en los que la propina fue superior al 20% del costo del viaje.

[ **EN** ]
- Develop a data science project following a structured workflow based on the Cookiecutter Data Science methodology.
- Build a machine learning model to predict taxi trips in New York City where the tip was greater than 20% of the fare.  


---

## Estructura del Proyecto // Project Structure

```
tip-prediction-nyc/
├── data/                  <- Raw & Processed data (included in .gitignore) 
├── models/                <- Saved models (joblib)
├── src/                   <- Source 
│   ├── data/              <- Download & Process dataset
│   ├── features/          <- Build features
│   ├── models/            <- Entrenamiento y evaluación del modelo
│   ├── visualization/     <- Plots and graphs
│   └── utils/             <- auxiliary functions
├── notebooks/             <- # Jupyter notebooks
├── requirements.txt       <- Project dependencies
└── README.md              <- This archive
└── .gitignore             <- Files ignored by Git
```

## Uso // Usage

1. Descarga los datos desde la fuente oficial // Download data from the official source  
   o usa la función  // or use the function
   `download_data(url, output_filename)` <- `src/data/dataset.py`. 

2. Procesa los datos // Process data
   ```python
   from data.dataset import load_and_process_dataset
   ```

3. Entrena el modelo // Train model
   ```python
   from models.train import train_model
   ```

4. Evalúa el modelo // Evaluate model
   ```python
   from models.predict import evaluate_f1
   ```

5. Genera visualizaciones // Generate visualization
   ```python
   from visualization.plots import plot_confusion_matrix, plot_f1_vs_threshold, ...
   ```

---

##  Acerca de // About
[**ES** ]
Los datos provienen de la NYC Taxi and Limousine Commission (TLC) y corresponden a los taxis amarillos.

El modelo es un Random Forest Classifier entrenado con los datos de enero de 2020. Luego se evalúa durante tres meses, utilizando F1-score como métrica principal de evaluación.

[ **EN** ]

The data come from the NYC Taxi and Limousine Commission (TLC) and relate to yellow taxi.

The model is a Random Forest Classifier trained on January 2020 data and then evaluated over three months  using F1-score as the main metric.