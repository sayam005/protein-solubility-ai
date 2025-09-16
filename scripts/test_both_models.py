import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_trained_models():
    """Load both trained models"""
    print("🤖 Loading trained models...")
    
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        scaler = joblib.load('models/data_scaler.pkl')
        
        print("✅ Random Forest model loaded")
        print("✅ Logistic Regression model loaded")
        print("✅ Data scaler loaded")
        
        return rf_model, lr_model, scaler
    
    except FileNotFoundError:
        print("❌ Models not found! Run 'python scripts/train_models.py' first")
        return None, None, None

def predict_protein_solubility(rf_model, lr_model, scaler, protein_data):
    """Predict solubility using both models"""
    
    # Convert to numpy array
    data_array = np.array([protein_data])
    
    # Random Forest prediction (doesn't need scaling)
    rf_prediction = rf_model.predict(data_array)[0]
    rf_probability = rf_model.predict_proba(data_array)[0][1]
    
    # Logistic Regression prediction (needs scaling)
    data_scaled = scaler.transform(data_array)
    lr_prediction = lr_model.predict(data_scaled)[0]
    lr_probability = lr_model.predict_proba(data_scaled)[0][1]
    
    return rf_prediction, rf_probability, lr_prediction, lr_probability

def test_individual_models():
    """Test Decision Tree only OR Logistic Regression only (for teacher questions)"""
    
    print("🎓 INDIVIDUAL MODEL TESTING")
    print("=" * 35)
    print("Testing single algorithms independently\n")
    
    # Load data
    print("📂 Loading dataset...")
    df = pd.read_csv('data/protein_dataset.csv')
    X = df[['molecular_weight', 'isoelectric_point', 'hydrophobicity', 
            'sequence_length', 'hydrophobic_aa_percent', 'charged_aa_percent']]
    y = df['solubility']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    while True:
        print("\nWhich individual model to test?")
        print("1. Decision Tree ONLY")
        print("2. Logistic Regression ONLY") 
        print("3. Compare individual models")
        print("4. Back to main menu")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            print("\n🌳 TESTING SINGLE DECISION TREE")
            print("-" * 35)
            
            # Train single Decision Tree
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
            dt_model.fit(X_train, y_train)
            
            # Test accuracy
            dt_pred = dt_model.predict(X_test)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            
            print(f"🎯 Decision Tree Accuracy: {dt_accuracy:.3f} ({dt_accuracy*100:.1f}%)")
            
            # Interactive testing for single model
            print(f"\n🧪 Test a protein with Decision Tree:")
            try:
                mol_weight = float(input("Molecular Weight (kDa): "))
                pi = float(input("Isoelectric Point: "))
                hydrophobicity = float(input("Hydrophobicity: "))
                length = int(input("Sequence Length: "))
                hydrophobic_aa = float(input("Hydrophobic AA %: "))
                charged_aa = float(input("Charged AA %: "))
                
                sample_protein = [mol_weight, pi, hydrophobicity, length, hydrophobic_aa, charged_aa]
                prediction = dt_model.predict([sample_protein])[0]
                probability = dt_model.predict_proba([sample_protein])[0][1]
                
                print(f"\n🌳 Decision Tree Result:")
                print(f"   Prediction: {'SOLUBLE' if prediction == 1 else 'INSOLUBLE'}")
                print(f"   Confidence: {probability:.1%}")
                
            except ValueError:
                print("❌ Invalid input!")
            
        elif choice == '2':
            print("\n📊 TESTING LOGISTIC REGRESSION ONLY")
            print("-" * 40)
            
            # Scale data for Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Logistic Regression
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            
            # Test accuracy
            lr_pred = lr_model.predict(X_test_scaled)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            
            print(f"🎯 Logistic Regression Accuracy: {lr_accuracy:.3f} ({lr_accuracy*100:.1f}%)")
            
            # Interactive testing for single model
            print(f"\n🧪 Test a protein with Logistic Regression:")
            try:
                mol_weight = float(input("Molecular Weight (kDa): "))
                pi = float(input("Isoelectric Point: "))
                hydrophobicity = float(input("Hydrophobicity: "))
                length = int(input("Sequence Length: "))
                hydrophobic_aa = float(input("Hydrophobic AA %: "))
                charged_aa = float(input("Charged AA %: "))
                
                sample_protein = [mol_weight, pi, hydrophobicity, length, hydrophobic_aa, charged_aa]
                sample_scaled = scaler.transform([sample_protein])
                prediction = lr_model.predict(sample_scaled)[0]
                probability = lr_model.predict_proba(sample_scaled)[0][1]
                
                print(f"\n📊 Logistic Regression Result:")
                print(f"   Prediction: {'SOLUBLE' if prediction == 1 else 'INSOLUBLE'}")
                print(f"   Confidence: {probability:.1%}")
                
            except ValueError:
                print("❌ Invalid input!")
            
        elif choice == '3':
            print("\n🏆 COMPARING INDIVIDUAL MODELS")
            print("-" * 35)
            
            # Decision Tree
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
            dt_model.fit(X_train, y_train)
            dt_pred = dt_model.predict(X_test)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            
            # Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            
            print(f"🌳 Decision Tree:       {dt_accuracy:.3f} ({dt_accuracy*100:.1f}%)")
            print(f"📊 Logistic Regression: {lr_accuracy:.3f} ({lr_accuracy*100:.1f}%)")
            
            if dt_accuracy > lr_accuracy:
                print(f"🥇 Decision Tree wins by {dt_accuracy - lr_accuracy:.3f}")
            else:
                print(f"🥇 Logistic Regression wins by {lr_accuracy - dt_accuracy:.3f}")
                
        elif choice == '4':
            break
        else:
            print("❌ Invalid choice!")

def test_both_models():
    """Interactive testing with both models (main feature)"""
    
    print("🧬 PFIZER PROTEIN SOLUBILITY PREDICTOR")
    print("=" * 50)
    print("Test protein solubility with both Random Forest and Logistic Regression\n")
    
    # Load models
    rf_model, lr_model, scaler = load_trained_models()
    if not all([rf_model, lr_model, scaler]):
        return
    
    # Show model performance
    try:
        import json
        with open('models/model_comparison.json', 'r') as f:
            results = json.load(f)
        
        print(f"📊 Model Performance:")
        print(f"   Random Forest: {results['random_forest_accuracy']:.3f} accuracy")
        print(f"   Logistic Regression: {results['logistic_regression_accuracy']:.3f} accuracy")
        print(f"   Winner: {results['winner']}\n")
    except:
        pass
    
    print("📋 Enter protein properties:")
    print("   Molecular Weight: 15-120 kDa")
    print("   Isoelectric Point: 4-11") 
    print("   Hydrophobicity: -2 to +2")
    print("   Sequence Length: 80-800 amino acids")
    print("   Hydrophobic AA %: 0-50%")
    print("   Charged AA %: 0-40%")
    print()
    
    while True:
        print("-" * 50)
        print("Enter protein properties (or 'quit' to exit):")
        
        try:
            mol_weight = input("Molecular Weight (kDa): ")
            if mol_weight.lower() == 'quit':
                break
            mol_weight = float(mol_weight)
            
            pi = float(input("Isoelectric Point: "))
            hydrophobicity = float(input("Hydrophobicity: "))
            length = int(input("Sequence Length: "))
            hydrophobic_aa = float(input("Hydrophobic AA %: "))
            charged_aa = float(input("Charged AA %: "))
            
            # Create protein data
            protein_data = [mol_weight, pi, hydrophobicity, length, hydrophobic_aa, charged_aa]
            
            # Get predictions from both models
            rf_pred, rf_prob, lr_pred, lr_prob = predict_protein_solubility(
                rf_model, lr_model, scaler, protein_data
            )
            
            # Display results
            print(f"\n🔬 PREDICTION RESULTS:")
            print("=" * 30)
            
            print(f"🌲 RANDOM FOREST:")
            print(f"   Prediction: {'SOLUBLE' if rf_pred == 1 else 'INSOLUBLE'}")
            print(f"   Confidence: {rf_prob:.1%}")
            
            print(f"\n📊 LOGISTIC REGRESSION:")
            print(f"   Prediction: {'SOLUBLE' if lr_pred == 1 else 'INSOLUBLE'}")
            print(f"   Confidence: {lr_prob:.1%}")
            
            # Agreement analysis
            if rf_pred == lr_pred:
                confidence_avg = (rf_prob + lr_prob) / 2
                print(f"\n✅ BOTH MODELS AGREE")
                print(f"🎯 Final Prediction: {'SOLUBLE' if rf_pred == 1 else 'INSOLUBLE'}")
                print(f"📈 Average Confidence: {confidence_avg:.1%}")
            else:
                print(f"\n⚠️  MODELS DISAGREE!")
                print(f"🤔 Recommendation: Review protein design")
                
                if rf_prob > lr_prob:
                    print(f"💡 Random Forest more confident ({rf_prob:.1%} vs {lr_prob:.1%})")
                else:
                    print(f"💡 Logistic Regression more confident ({lr_prob:.1%} vs {rf_prob:.1%})")
            
            print()
            
        except ValueError:
            print("❌ Please enter valid numbers!")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for using the Pfizer protein solubility predictor!")

def main():
    """Main menu - clean and professional"""
    print("🤖 PFIZER PROTEIN SOLUBILITY PREDICTION SYSTEM")
    print("=" * 55)
    print("Machine Learning Models for E. coli Expression Prediction")
    
    while True:
        print("\nChoose testing mode:")
        print("1. Test both models together (recommended)")
        print("2. Test individual models separately")
        print("3. Quit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            test_both_models()
        elif choice == '2':
            test_individual_models()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter 1-3.")

if __name__ == "__main__":
    main()