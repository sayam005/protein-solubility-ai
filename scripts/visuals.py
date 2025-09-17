import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

def create_presentation_visuals():
    """Create all visuals needed for 5-minute presentation"""
    
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
    
    # Create all presentation visuals
    create_flow_diagram()                                          # 1. Project workflow
    create_accuracy_bar_chart(rf_model, lr_model, scaler, X_test, y_test)  # 2. Model comparison
    create_roc_comparison(rf_model, lr_model, scaler, X_test, y_test)       # 3. ROC curves
    create_confusion_matrices(rf_model, lr_model, scaler, X_test, y_test)   # 4. Error analysis
    
    print("\n✅ All presentation visuals created in outputs/ folder")
    print("📂 Files for 5-minute presentation:")
    print("   1. project_workflow.png (methodology)")
    print("   2. accuracy_comparison.png (performance)")
    print("   3. roc_curves.png (model quality)")
    print("   4. confusion_matrices.png (error analysis)")

def create_flow_diagram():
    """Create simple project workflow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Simple workflow boxes
    boxes = [
        ("Data Generation\n1,500 Proteins", (2, 7), "#FFE6E6"),
        ("Feature Selection\n6 Properties", (2, 5.5), "#E6F3FF"),
        ("Train-Test Split\n80/20", (2, 4), "#F0E6FF"),
        ("Model Training\nRF + LR", (6, 4), "#FFE6F0"),
        ("Performance Evaluation\nAccuracy + ROC", (10, 4), "#E6FFFF"),
        ("Interactive Testing\nBoth Models", (6, 2.5), "#E6F5F5")
    ]
    
    # Draw boxes
    for text, (x, y), color in boxes:
        rect = patches.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                               linewidth=2, edgecolor='black', 
                               facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((2, 6.6), (2, 5.9)),    # Generation -> Features
        ((2, 5.1), (2, 4.4)),    # Features -> Split
        ((2.8, 4), (5.2, 4)),    # Split -> Training
        ((6.8, 4), (9.2, 4)),    # Training -> Evaluation
        ((6, 3.6), (6, 2.9))     # Training -> Testing
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Protein Solubility AI - Project Workflow', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('outputs/project_workflow.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: project_workflow.png")
    plt.close()

def create_accuracy_bar_chart(rf_model, lr_model, scaler, X_test, y_test):
    """Simple accuracy comparison"""
    
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(scaler.transform(X_test))
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    plt.figure(figsize=(8, 6))
    models = ['Random Forest', 'Logistic Regression']
    accuracies = [rf_accuracy, lr_accuracy]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, width=0.6)
    
    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.ylim(0, 1)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    winner = 'Random Forest' if rf_accuracy > lr_accuracy else 'Logistic Regression'
    plt.text(0.5, 0.85, f'🏆 Winner: {winner}', 
             transform=plt.gca().transAxes, ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: accuracy_comparison.png")
    plt.close()

def create_roc_comparison(rf_model, lr_model, scaler, X_test, y_test):
    """ROC curve comparison"""
    
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    lr_prob = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    
    rf_auc = auc(rf_fpr, rf_tpr)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, color='#2E86AB', lw=3, 
             label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(lr_fpr, lr_tpr, color='#A23B72', lw=3, 
             label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: roc_curves.png")
    plt.close()

def create_confusion_matrices(rf_model, lr_model, scaler, X_test, y_test):
    """Confusion matrices for error analysis"""
    
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(scaler.transform(X_test))
    
    rf_cm = confusion_matrix(y_test, rf_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Confusion Matrices - Error Analysis', fontsize=14, fontweight='bold')
    
    # Random Forest
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Insoluble', 'Soluble'],
                yticklabels=['Insoluble', 'Soluble'])
    ax1.set_title('Random Forest')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Logistic Regression
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Insoluble', 'Soluble'], 
                yticklabels=['Insoluble', 'Soluble'])
    ax2.set_title('Logistic Regression')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrices.png")
    plt.close()

if __name__ == "__main__":
    create_presentation_visuals()