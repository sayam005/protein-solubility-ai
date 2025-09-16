import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

def load_and_explore_data():
    """Load the protein dataset and show basic info"""
    print("📂 Loading protein dataset...")
    
    df = pd.read_csv('data/protein_dataset.csv')
    
    print(f"✅ Loaded {len(df)} proteins")
    print(f"📊 Total features: {len(df.columns)-1}")
    print(f"🎯 Target: solubility (1=soluble, 0=insoluble)")
    
    # Show class distribution
    soluble_count = df['solubility'].sum()
    print(f"🟢 Soluble: {soluble_count} ({soluble_count/len(df)*100:.1f}%)")
    print(f"🔴 Insoluble: {len(df)-soluble_count} ({(len(df)-soluble_count)/len(df)*100:.1f}%)")
    
    return df

def prepare_data_for_training(df):
    """Split data into features and target, then train/test sets"""
    print("\n🔧 Preparing data for machine learning...")
    
    # Use main physicochemical features + amino acid summaries
    feature_columns = ['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
                      'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']
    
    X = df[feature_columns]
    y = df['solubility']
    
    print(f"📈 Features selected: {len(feature_columns)} features")
    print(f"   Core features: molecular_weight, isoelectric_point, hydrophobicity, sequence_length")
    print(f"   Amino acid features: hydrophobic_aa_percent, charged_aa_percent")
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🚆 Training set: {len(X_train)} proteins")
    print(f"🧪 Test set: {len(X_test)} proteins")
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n🌲 Training Random Forest...")
    print("   (Random Forest doesn't need data scaling)")
    
    # Create and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,        # 100 decision trees
        random_state=42,         # For reproducible results
        max_depth=10            # Prevent overfitting
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:, 1]  # Probability of being soluble
    
    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    print(f"✅ Random Forest trained!")
    print(f"🎯 Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
    
    return rf_model, rf_predictions, rf_probabilities, rf_accuracy

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model"""
    print("\n📊 Training Logistic Regression...")
    print("   (Logistic Regression needs data scaling)")
    
    # Scale the data (IMPORTANT for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   ⚖️  Data scaled to standard range")
    
    # Create and train Logistic Regression
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000           # Ensure convergence
    )
    
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions on scaled test data
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate accuracy
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    
    print(f"✅ Logistic Regression trained!")
    print(f"🎯 Accuracy: {lr_accuracy:.3f} ({lr_accuracy*100:.1f}%)")
    
    return lr_model, scaler, lr_predictions, lr_probabilities, lr_accuracy

def compare_models(rf_accuracy, lr_accuracy, y_test, rf_pred, lr_pred):
    """Compare both models and determine winner"""
    print(f"\n🏆 MODEL COMPARISON:")
    print("=" * 40)
    print(f"Random Forest Accuracy:      {rf_accuracy:.3f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
    print(f"Difference:                  {abs(rf_accuracy - lr_accuracy):.3f}")
    
    if rf_accuracy > lr_accuracy:
        winner = "Random Forest"
        print(f"\n🥇 WINNER: Random Forest")
        print(f"   Random Forest beats Logistic Regression by {rf_accuracy - lr_accuracy:.3f}")
    elif lr_accuracy > rf_accuracy:
        winner = "Logistic Regression"
        print(f"\n🥇 WINNER: Logistic Regression")
        print(f"   Logistic Regression beats Random Forest by {lr_accuracy - rf_accuracy:.3f}")
    else:
        winner = "Tie"
        print(f"\n🤝 IT'S A TIE!")
    
    # Check agreement between models
    agreement = (rf_pred == lr_pred).mean()
    print(f"\n🤝 Model Agreement: {agreement:.1%}")
    print(f"   (How often both models make the same prediction)")
    
    return winner

def save_models_and_results(rf_model, lr_model, scaler, rf_accuracy, lr_accuracy, winner):
    """Save both trained models and results"""
    print(f"\n💾 Saving models and results...")
    
    # Save models
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    joblib.dump(scaler, 'models/data_scaler.pkl')
    
    # Save comparison results
    results = {
        'random_forest_accuracy': float(rf_accuracy),
        'logistic_regression_accuracy': float(lr_accuracy),
        'winner': winner,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': 1500,
        'test_size': 300
    }
    
    with open('models/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Random Forest saved: models/random_forest_model.pkl")
    print(f"✅ Logistic Regression saved: models/logistic_regression_model.pkl")
    print(f"✅ Data scaler saved: models/data_scaler.pkl")
    print(f"✅ Results saved: models/model_comparison.json")

def main():
    """Main training pipeline"""
    print("🤖 PFIZER PROTEIN SOLUBILITY - MODEL TRAINER")
    print("=" * 50)
    print("Training both Random Forest and Logistic Regression models")
    print("NOW WITH AMINO ACID COMPOSITION FEATURES!")
    
    # Step 1: Load data
    df = load_and_explore_data()
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, features = prepare_data_for_training(df)
    
    # Step 3: Train Random Forest
    rf_model, rf_pred, rf_prob, rf_accuracy = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Step 4: Train Logistic Regression
    lr_model, scaler, lr_pred, lr_prob, lr_accuracy = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Step 5: Compare models
    winner = compare_models(rf_accuracy, lr_accuracy, y_test, rf_pred, lr_pred)
    
    # Step 6: Save everything
    save_models_and_results(rf_model, lr_model, scaler, rf_accuracy, lr_accuracy, winner)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"✅ Dataset now includes amino acid composition as required!")
    print(f"🎯 Next step: Create test script to use both models")

if __name__ == "__main__":
    main()