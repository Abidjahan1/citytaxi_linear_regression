# citytaxi_linear_regression

# 🚕 Taxi Fare Prediction with TensorFlow

This project trains regression models to predict taxi fares based on trip features such as distance (`TRIP_MILES`) and duration (`TRIP_MINUTES`). It uses TensorFlow to build and evaluate models, and includes tools to make and display predictions.

---
📊 Dataset Overview

📂 chicago_taxi_train.csv

Key Features used:

TRIP_MILES

TRIP_SECONDS

FARE

COMPANY

PAYMENT_TYPE

TIP_RATE

🚀 Features & Functionality

✅ Load and clean real-world taxi data

✅ Analyze dataset statistics (e.g., max fare, average distance, missing data)

✅ Explore correlations using a pairplot and correlation matrix

✅ Build a simple linear regression model with TensorFlow/Keras

✅ Visualize:

Model prediction surface (2D/3D)

Loss curve (RMSE vs Epochs)

✅ Print model weights, bias, and training progress


## 📊 Model Training

### ✅ Experiment 1: One Feature (`TRIP_MILES`)

A simple model is trained using just the trip distance:

```python
features = ['TRIP_MILES']
label = 'FARE'
```

**Hyperparameters:**
- Learning Rate: `0.001`
- Epochs: `20`
- Batch Size: `50`

Model is trained using:

```python
model_1 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)
```

---

### ✅ Experiment 2: Two Features (`TRIP_MILES`, `TRIP_MINUTES`)

We enhance the model by adding trip duration (converted from seconds to minutes):

```python
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS'] / 60
features = ['TRIP_MILES', 'TRIP_MINUTES']
label = 'FARE'
```

Same hyperparameters as above. Trained with:

```python
model_2 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)
```

---

## 🧠 Making Predictions

Predictions are generated using:

```python
output = predict_fare(model_2, training_df, features, label)
show_predictions(output)
```

The prediction output includes:
- `PREDICTED_FARE`: Model's output
- `OBSERVED_FARE`: Actual fare
- `L1_LOSS`: Absolute difference between predicted and observed
- Feature values used in prediction

### Sample Output

```
--------------------------------------------------------------------------------
|                                 PREDICTIONS                                 |
--------------------------------------------------------------------------------
  PREDICTED_FARE  OBSERVED_FARE  L1_LOSS  TRIP_MILES  TRIP_MINUTES
0        $8.55          $9.00     $0.45        2.10           12.0
...
```

---

## 🛠️ Key Functions

- `run_experiment()`: Trains the model on the provided dataset.
- `predict_fare()`: Generates fare predictions for a sample batch.
- `show_predictions()`: Displays predictions in a formatted table.

---

## 📦 Requirements

- Python 3.x
- Pandas
- NumPy
- TensorFlow

Install with:

```bash
pip install pandas numpy tensorflow
```

---

## 📁 Project Structure (Sample)

```
.
├── taxi_fare_prediction.ipynb
├── README.md
└── data/
    └── taxi_trips.csv
```

---

## 📌 Author

Built as part of a machine learning project on taxi fare prediction.
