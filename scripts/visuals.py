import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split

def create_presentation_visuals():
    """Create simple, clean visuals for presentation"""
    
    print("📊 CREATING PRESENTATION VISUALS")
    print("=" * 40)
    
    # Load data
    df = pd.read_csv('data/protein_dataset.csv')
    X = df[['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
            'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']]
    y = df['solubility']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load models
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/data_scaler.pkl')
    
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Accuracy Comparison Bar Chart
    create_accuracy_bar_chart(rf_model, lr_model, scaler, X_test, y_test)
    
    # 2. ROC Curve Comparison
    create_roc_comparison(rf_model, lr_model, scaler, X_test, y_test)
    
    # 3. Dataset Overview
    create_dataset_summary(df)
    
    # 4. Preprocessing Summary (text output)
    show_preprocessing_summary(df)
    
    print("\n✅ All visuals created in outputs/ folder")
    print("📂 Files created:")
    print("   - accuracy_comparison.png")
    print("   - roc_curves.png") 
    print("   - dataset_summary.png")

def create_accuracy_bar_chart(rf_model, lr_model, scaler, X_test, y_test):
    """Simple accuracy comparison bar chart"""
    
    # Get predictions and accuracies
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(scaler.transform(X_test))
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    models = ['Random Forest', 'Logistic Regression']
    accuracies = [rf_accuracy, lr_accuracy]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, width=0.6)
    
    # Add percentage labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.ylim(0, 1)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Model Performance Comparison\nProtein Solubility Prediction', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add winner text
    winner = 'Random Forest' if rf_accuracy > lr_accuracy else 'Logistic Regression'
    difference = abs(rf_accuracy - lr_accuracy)
    plt.text(0.5, 0.85, f'🏆 Winner: {winner}\nDifference: {difference:.1%}', 
             transform=plt.gca().transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: accuracy_comparison.png")
    plt.close()

def create_roc_comparison(rf_model, lr_model, scaler, X_test, y_test):
    """Combined ROC curve comparison"""
    
    # Get probabilities
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    lr_prob = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    
    # Calculate ROC curves
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    
    rf_auc = auc(rf_fpr, rf_tpr)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    # Create ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, color='#2E86AB', lw=3, 
             label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(lr_fpr, lr_tpr, color='#A23B72', lw=3, 
             label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison\nProtein Solubility Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: roc_curves.png")
    plt.close()

def create_dataset_summary(df):
    """Dataset overview visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pfizer Protein Dataset - Overview', fontsize=16, fontweight='bold')
    
    # 1. Solubility Distribution (Pie Chart)
    solubility_counts = df['solubility'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(solubility_counts.values, labels=['Insoluble', 'Soluble'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Solubility Distribution', fontweight='bold')
    
    # 2. Molecular Weight Distribution
    soluble_mw = df[df['solubility'] == 1]['molecular_weight']
    insoluble_mw = df[df['solubility'] == 0]['molecular_weight']
    
    ax2.hist(insoluble_mw, alpha=0.7, label='Insoluble', color='#FF6B6B', bins=20)
    ax2.hist(soluble_mw, alpha=0.7, label='Soluble', color='#4ECDC4', bins=20)
    ax2.set_xlabel('Molecular Weight (kDa)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Molecular Weight Distribution')
    ax2.legend()
    
    # 3. Hydrophobicity Distribution
    soluble_hydro = df[df['solubility'] == 1]['hydrophobicity']
    insoluble_hydro = df[df['solubility'] == 0]['hydrophobicity']
    
    ax3.hist(insoluble_hydro, alpha=0.7, label='Insoluble', color='#FF6B6B', bins=20)
    ax3.hist(soluble_hydro, alpha=0.7, label='Soluble', color='#4ECDC4', bins=20)
    ax3.set_xlabel('Hydrophobicity Index')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Hydrophobicity Distribution')
    ax3.legend()
    
    # 4. Feature Correlation with Solubility
    features = ['Molecular\nWeight', 'Isoelectric\nPoint', 'Hydrophobicity', 
                'Sequence\nLength', 'Hydrophobic\nAA%', 'Charged\nAA%']
    correlations = df[['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
                      'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']].corrwith(df['solubility']).abs()
    
    bars = ax4.bar(features, correlations, color='#95E1D3', alpha=0.8)
    ax4.set_ylabel('Correlation with Solubility')
    ax4.set_title('Feature Importance (Correlation)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{corr:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/dataset_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: dataset_summary.png")
    plt.close()

def show_preprocessing_summary(df):
    """Show preprocessing info (for presentation slides)"""
    
    print(f"\n📋 PREPROCESSING SUMMARY (for presentation):")
    print("=" * 50)
    print(f"📊 Dataset Size: {len(df)} proteins")
    print(f"🎯 Target Classes: {df['solubility'].value_counts().values}")
    print(f"   - Soluble: {df['solubility'].sum()} ({df['solubility'].mean():.1%})")
    print(f"   - Insoluble: {len(df) - df['solubility'].sum()} ({1-df['solubility'].mean():.1%})")
    
    print(f"\n🔧 Features Used (6 total):")
    features = ['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
                'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature.replace('_', ' ').title()}")
    
    print(f"\n✅ Preprocessing Steps:")
    print(f"   ✓ No missing values (synthetic data)")
    print(f"   ✓ 80/20 train-test split")
    print(f"   ✓ StandardScaler for Logistic Regression")
    print(f"   ✓ Stratified sampling (maintains class balance)")
    
    print(f"\n📈 Data Ranges:")
    numerical_features = ['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
                         'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']
    
    for feature in numerical_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        mean_val = df[feature].mean()
        print(f"   {feature}: {min_val:.1f} - {max_val:.1f} (avg: {mean_val:.1f})")

if __name__ == "__main__":
    create_presentation_visuals()