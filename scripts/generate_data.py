import pandas as pd
import numpy as np
import random
import os

def generate_amino_acid_composition():
    """Generate realistic amino acid percentages that sum to 100%"""
    # Key amino acids that affect solubility
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Generate random percentages
    raw_percentages = [random.uniform(1, 10) for _ in range(20)]
    total = sum(raw_percentages)
    
    # Normalize to sum to 100%
    normalized = [(p / total) * 100 for p in raw_percentages]
    return {aa: round(pct, 2) for aa, pct in zip(amino_acids, normalized)}

def create_protein_dataset():
    """
    Creates a realistic protein dataset for solubility prediction.
    This simulates what Pfizer would use for drug development.
    """
    
    print("🧬 Creating protein dataset for Pfizer project...")
    print("   Generating 1500 synthetic proteins with realistic properties")
    
    # Set seeds for reproducible results (important for academic work)
    random.seed(42)
    np.random.seed(42)
    
    # Create necessary folders
    print("📁 Creating project folders...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    proteins = []
    
    print("⚗️  Generating protein properties...")
    for i in range(1500):
        protein_id = f'PROT_{i+1:04d}'
        
        # Generate amino acid composition first
        aa_composition = generate_amino_acid_composition()
        
        # Generate realistic physicochemical properties
        molecular_weight = round(random.uniform(15.0, 120.0), 2)    # kDa
        isoelectric_point = round(random.uniform(4.0, 11.0), 2)     # pI
        hydrophobicity = round(random.uniform(-2.0, 2.0), 3)        # GRAVY score
        sequence_length = random.randint(80, 800)                   # amino acids
        
        # Calculate derived features from amino acid composition
        hydrophobic_residues = (aa_composition['A'] + aa_composition['I'] + 
                               aa_composition['L'] + aa_composition['V'] + 
                               aa_composition['F'] + aa_composition['W'])
        
        charged_residues = (aa_composition['R'] + aa_composition['K'] + 
                           aa_composition['D'] + aa_composition['E'])
        
        # Determine solubility based on known biochemical rules
        solubility_score = 0
        
        # Rule 1: Lower hydrophobicity = more soluble
        if hydrophobicity < -0.5:
            solubility_score += 2
        elif hydrophobicity < 0:
            solubility_score += 1
        elif hydrophobicity > 1.0:
            solubility_score -= 2
        
        # Rule 2: Moderate size is better for solubility
        if molecular_weight < 60:
            solubility_score += 1
        elif molecular_weight > 100:
            solubility_score -= 1
        
        # Rule 3: Moderate length is better
        if 100 <= sequence_length <= 400:
            solubility_score += 1
        elif sequence_length > 600:
            solubility_score -= 1
        
        # Rule 4: Amino acid composition effects
        if hydrophobic_residues > 35:  # Too hydrophobic
            solubility_score -= 1
        if charged_residues > 25:      # High charge helps
            solubility_score += 1
        
        # Rule 5: Add some biological randomness
        solubility_score += random.choice([-1, 0, 1])
        
        # Final decision: positive score = soluble
        is_soluble = 1 if solubility_score > 0 else 0
        
        # Store protein data
        protein = {
            'protein_id': protein_id,
            'molecular_weight': molecular_weight,
            'isoelectric_point': isoelectric_point,
            'hydrophobicity': hydrophobicity,
            'sequence_length': sequence_length,
            'hydrophobic_aa_percent': round(hydrophobic_residues, 2),
            'charged_aa_percent': round(charged_residues, 2),
            'solubility': is_soluble
        }
        
        # Add individual amino acid percentages
        for aa, percentage in aa_composition.items():
            protein[f'aa_{aa}'] = percentage
        
        proteins.append(protein)
        
        # Show progress every 300 proteins
        if (i + 1) % 300 == 0:
            print(f"   Generated {i + 1} proteins...")
    
    # Create DataFrame and save
    df = pd.DataFrame(proteins)
    df.to_csv('data/protein_dataset.csv', index=False)
    
    # Show summary statistics
    soluble_count = df['solubility'].sum()
    insoluble_count = len(df) - soluble_count
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📊 DATASET SUMMARY:")
    print(f"   Total proteins: {len(df)}")
    print(f"   Features: {len(df.columns)-1} (including amino acid composition)")
    print(f"   Soluble proteins: {soluble_count} ({soluble_count/len(df)*100:.1f}%)")
    print(f"   Insoluble proteins: {insoluble_count} ({insoluble_count/len(df)*100:.1f}%)")
    print(f"   Saved to: data/protein_dataset.csv")
    
    # Show first few examples
    print(f"\n🔬 Sample protein (first 8 columns):")
    print(df.iloc[:3, :8])
    
    print(f"\n🎯 Next step: Run 'python scripts/train_models.py'")
    
    return df

if __name__ == "__main__":
    create_protein_dataset()