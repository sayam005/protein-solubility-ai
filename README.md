# 🧬 Protein Solubility AI - Pfizer Project

Machine learning models to predict therapeutic protein solubility in E. coli expression systems, helping Pfizer optimize drug development pipelines.

## 🎯 Project Overview

This project develops supervised classification models to predict whether therapeutic proteins will be **soluble** or **insoluble** when expressed in E. coli bacteria. Early solubility prediction can save pharmaceutical companies millions in failed experimental costs.

## 📊 Dataset

- **Size**: 1,500 synthetic proteins with realistic biological properties
- **Features**: 6 physicochemical properties based on biochemistry literature
- **Target**: Binary classification (Soluble: 1, Insoluble: 0)
- **Distribution**: 56.5% insoluble, 43.5% soluble (reflects real-world challenges)

## 🔬 Features Used

| Feature | Range | Biological Significance |
|---------|-------|------------------------|
| **Molecular Weight** | 15-120 kDa | Larger proteins harder to fold correctly |
| **Isoelectric Point** | 4-11 pH | Extreme pI values cause instability |
| **Hydrophobicity Index** | -2 to +2 | Water-hating proteins tend to aggregate |
| **Sequence Length** | 80-800 amino acids | Longer sequences more complex to fold |
| **Hydrophobic AA %** | 15-50% | High hydrophobic content → aggregation |
| **Charged AA %** | 8-35% | Charged residues improve water solubility |

## 🤖 Models Implemented

### **Random Forest Classifier**
- **Accuracy**: 87.0% 🏆
- **AUC Score**: 0.945
- **Ensemble of 100 decision trees**
- **Handles non-linear protein interactions**

### **Logistic Regression**  
- **Accuracy**: 84.7%
- **AUC Score**: 0.924
- **Linear classification with feature scaling**
- **Highly interpretable coefficients**

### **Model Agreement**: 90.3% consensus rate

## 🛠️ Installation & Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd protein-solubility-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib joblib
   ```

## 🚀 Usage

### **Generate Synthetic Dataset**
```bash
python scripts/generate_data.py
```

### **Train Both Models**
```bash
python scripts/train_models.py
```

### **Create Visualizations**
```bash
python scripts/visuals.py
```

### **Interactive Testing**
```bash
python scripts/test_both_models.py
```

## 📈 Results & Performance

### **Model Comparison**
- Random Forest outperforms Logistic Regression by 2.3%
- Both models achieve excellent AUC scores (>0.92)
- High agreement rate demonstrates model reliability

### **Key Findings**
- **Hydrophobicity** is the strongest predictor (correlation: 0.63)
- **Molecular weight** and **sequence length** significantly impact solubility
- **Charged amino acids** improve solubility as expected from biochemistry

## 📊 Visualizations

The project generates three key visualizations:

1. **`accuracy_comparison.png`** - Model performance bar chart
2. **`roc_curves.png`** - ROC curve analysis with AUC scores  
3. **`dataset_summary.png`** - Dataset overview and feature distributions

## 🔬 Technical Implementation

### **Data Preprocessing**
- ✅ No missing values (synthetic dataset advantage)
- ✅ Feature scaling with StandardScaler for Logistic Regression
- ✅ 80/20 train-test split with stratification
- ✅ Biologically realistic parameter ranges

### **Model Training**
- ✅ Random Forest: 100 estimators, max_depth=10
- ✅ Logistic Regression: L2 regularization, max_iter=1000
- ✅ Cross-validation with stratified sampling
- ✅ Model persistence with joblib

## 🧪 Example Predictions

### **Highly Soluble Protein**
```
MW: 35 kDa, pI: 7.0, Hydrophobicity: -1.2
→ Both models predict: SOLUBLE (>85% confidence)
```

### **Clearly Insoluble Protein**
```
MW: 95 kDa, pI: 10.2, Hydrophobicity: 1.6  
→ Both models predict: INSOLUBLE (>90% confidence)
```

## 📂 Project Structure

```
protein-solubility-ai/
├── scripts/
│   ├── generate_data.py      # Dataset creation
│   ├── train_models.py       # Model training
│   ├── test_both_models.py   # Interactive testing
│   └── visuals.py           # Visualization generation
├── data/
│   └── protein_dataset.csv   # Synthetic protein data
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── data_scaler.pkl
│   └── model_comparison.json
├── outputs/
│   ├── accuracy_comparison.png
│   ├── roc_curves.png
│   └── dataset_summary.png
└── README.md 