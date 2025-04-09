import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        """Preprocess the student data."""
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values
        df = df.fillna(df.mode().iloc[0])
        
        # Encode categorical variables
        categorical_cols = ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                          'reason', 'guardian', 'schoolsup', 'famsup', 'paid',
                          'activities', 'nursery', 'higher', 'internet', 'romantic']
        
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Create derived features
        df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
        df['alcohol_consumption'] = (df['Dalc'] + df['Walc']) / 2
        
        # Scale numerical features
        numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                         'failures', 'freetime', 'goout', 'Dalc', 'Walc',
                         'health', 'absences', 'G1', 'G2', 'parent_edu_avg',
                         'alcohol_consumption']
        
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
