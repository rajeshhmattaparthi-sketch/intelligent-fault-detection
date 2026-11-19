# Intelligent Fault Detection

A simple machine-learning project that detects and classifies faults in a machine
(e.g. an electric motor) from sensor readings.

## Features
- Simulated sensor data (vibration, temperature, current, voltage, RPM)
- Anomaly detection with Isolation Forest ("is something wrong?")
- Fault classification with Random Forest (Normal, Bearing Fault, Overheating, Overload)
- Evaluation report, confusion matrix and feature importance plots
- Saved model (`fault_model.joblib`) and a live prediction demo

## Setup
```
pip install -r requirements.txt
```

## Run
```
python fault_detection.py
```

## Using your own data
Replace `df = generate_data()` in `fault_detection.py` with:
```
df = pd.read_csv("your_data.csv")
```
The CSV needs these columns: `vibration, temperature, current, voltage, rpm, fault`
(fault: 0 = Normal, 1 = Bearing Fault, 2 = Overheating, 3 = Overload).

## Output files
- `sensor_data.csv` - generated dataset
- `confusion_matrix.png`, `feature_importance.png` - plots
- `fault_model.joblib` - trained models
