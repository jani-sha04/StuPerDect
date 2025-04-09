import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_student_data(filepath):
    """Load student data from CSV file."""
    return pd.read_csv(filepath)

def plot_feature_importance(feature_names, importance_scores, title="Feature Importance"):
    """Plot feature importance scores."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_prediction_vs_actual(y_true, y_pred, title="Predicted vs Actual Grades"):
    """Create scatter plot of predicted vs actual grades."""
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'Actual Grades', 'y': 'Predicted Grades'},
        title=title
    )
    fig.add_shape(
        type='line',
        x0=min(y_true),
        y0=min(y_true),
        x1=max(y_true),
        y1=max(y_true),
        line=dict(color='red', dash='dash')
    )
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap for numerical features."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    return plt.gcf()

def generate_performance_report(metrics, model_name):
    """Generate a formatted performance report."""
    report = f"""
    Model Performance Report ({model_name})
    =====================================
    RMSE: {metrics['rmse']:.3f}
    MAE:  {metrics['mae']:.3f}
    R²:   {metrics['r2']:.3f}
    """
    return report
