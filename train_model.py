"""
Training Script Runner
Execute this to train and save the machine learning model.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Change to project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import and run the training script
if __name__ == "__main__":
    print("Starting model training...")
    print("=" * 50)
    
    # Execute the training script
    exec(open('src/ml/train.py').read())
    
    print("=" * 50)
    print("Training completed!")
