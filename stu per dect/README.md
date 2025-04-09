# Student Performance Predictor

A machine learning system that predicts student academic performance based on various demographic, social, and academic factors.

## Features
- Comprehensive data preprocessing
- Multiple ML models (Linear, Random Forest, SVM)
- Feature importance analysis
- Interactive web interface
- Performance visualization

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
streamlit run app.py
```

## Project Structure
- `model.py`: Core ML model implementation
- `preprocess.py`: Data preprocessing utilities
- `utils.py`: Helper functions
- `app.py`: Streamlit web interface
- `notebooks/`: Jupyter notebooks for analysis

## Models
- Linear Regression
- Random Forest Regressor
- Support Vector Regression

## Performance Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
