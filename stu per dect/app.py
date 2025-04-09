import streamlit as st
import pandas as pd
import numpy as np
from model import StudentPerformancePredictor
from preprocess import DataPreprocessor
from utils import plot_feature_importance, plot_prediction_vs_actual, create_correlation_heatmap

def main():
    st.title("Student Performance Predictor")
    st.write("""
    This application predicts student performance based on various demographic,
    social, and academic factors.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your student data CSV", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess data
        data = pd.read_csv(uploaded_file)
        preprocessor = DataPreprocessor()
        
        # Display raw data
        st.subheader("Raw Data Preview")
        st.write(data.head())
        
        # Preprocess data
        processed_data = preprocessor.preprocess_data(data)
        
        # Split features and target
        X = processed_data.drop('G3', axis=1)
        y = processed_data['G3']
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Train model
        model = StudentPerformancePredictor()
        
        with st.spinner('Training models...'):
            model.train(X_train, y_train, X_val, y_val)
            metrics = model.evaluate(X_test, y_test)
        
        # Display results
        st.subheader("Model Performance")
        st.write(f"Best Model: {model.best_model_name}")
        st.write(f"R² Score: {metrics['r2']:.3f}")
        st.write(f"RMSE: {metrics['rmse']:.3f}")
        st.write(f"MAE: {metrics['mae']:.3f}")
        
        # Feature importance plot
        if model.feature_importance is not None:
            st.subheader("Feature Importance")
            fig = plot_feature_importance(X.columns, model.feature_importance)
            st.pyplot(fig)
        
        # Predictions vs Actual plot
        st.subheader("Predictions vs Actual")
        y_pred = model.predict(X_test)
        fig = plot_prediction_vs_actual(y_test, y_pred)
        st.plotly_chart(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        fig = create_correlation_heatmap(data.select_dtypes(include=[np.number]))
        st.pyplot(fig)
        
        # Save model
        if st.button('Save Model'):
            model.save_model('student_performance_model.joblib')
            st.success('Model saved successfully!')

if __name__ == '__main__':
    main()
