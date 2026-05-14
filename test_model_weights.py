'''
this file examines the saved cvxNN as a sanity check
and its saved attributes (checks weights, dimensions compatible, ADMM solution variables)

example output:
Model loaded successfully
Model has theta1: True
Model has theta2: True
theta1 shape: (46080, 20)
theta1 min/max/mean: -0.013659/0.011700/0.000000
theta1 non-zero elements: 921600/921600 (100.00%)
theta2 shape: (20,)
theta2 min/max/mean: -0.137282/0.137303/-0.000006
theta1 and theta2 dimensions are compatible
Model has ADMM solution variable u: True
Model has ADMM solution variable v: True
Model has ADMM solution variable s: True

Model weights are stored and accessible
'''

import pickle
import numpy as np
import os

def test_model_weights(model_path):
    """
    Test if the model has the theta1 and theta2 weights properly stored.
    
    Args:
        model_path: Path to the trained model pickle file
    """
    print(f"Testing model at: {model_path}")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return False
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Check if theta1 and theta2 exist
    has_theta1 = hasattr(model, 'theta1')
    has_theta2 = hasattr(model, 'theta2')
    
    print(f"Model has theta1: {has_theta1}")
    print(f"Model has theta2: {has_theta2}")
    
    if has_theta1 and has_theta2:
        # Check theta1
        if model.theta1 is not None:
            print(f"theta1 shape: {model.theta1.shape}")
            print(f"theta1 min/max/mean: {np.min(model.theta1):.6f}/{np.max(model.theta1):.6f}/{np.mean(model.theta1):.6f}")
            print(f"theta1 non-zero elements: {np.count_nonzero(model.theta1)}/{model.theta1.size} ({np.count_nonzero(model.theta1)/model.theta1.size*100:.2f}%)")
        else:
            print("theta1 is None")
            
        # Check theta2
        if model.theta2 is not None:
            print(f"theta2 shape: {model.theta2.shape}")
            print(f"theta2 min/max/mean: {np.min(model.theta2):.6f}/{np.max(model.theta2):.6f}/{np.mean(model.theta2):.6f}")
        else:
            print("theta2 is None")
        
        # Check theta1 and theta2 compatibility
        if model.theta1 is not None and model.theta2 is not None:
            if model.theta1.shape[1] == model.theta2.shape[0]:
                print("theta1 and theta2 dimensions are compatible")
            else:
                print(f"Dimension mismatch: theta1 output dim ({model.theta1.shape[1]}) != theta2 input dim ({model.theta2.shape[0]})")
        
        # Check presence of ADMM variables
        has_u = hasattr(model, 'u') and model.u is not None
        has_v = hasattr(model, 'v') and model.v is not None
        has_s = hasattr(model, 's') and model.s is not None
        
        print(f"Model has ADMM solution variable u: {has_u}")
        print(f"Model has ADMM solution variable v: {has_v}")
        print(f"Model has ADMM solution variable s: {has_s}")
        
        # Print summary
        if model.theta1 is not None and model.theta2 is not None:
            print("\n Model weights are properly stored and accessible")
            return True
        else:
            print("\n Model weights are not properly stored")
            return False
    else:
        print("\n Model does not have both theta1 and theta2 attributes")
        return False

if __name__ == "__main__":
    # input model path
    #model_path = "/home/miria/CVXDPO/320_trained_cvxmlp_custom/custom_trained_cvx_mlp.pkl"
    #model_path = "/home/miria/CVXDPO/320_trained_cvxmlp_gpt2_imdb/gpt2_imdb_trained_cvx_mlp.pkl"
    model_path = "/home/miria/CVXDPO/cvxNN_trained_gpt2_attn_ultra/gpt2_attn_ultra_trained_cvx_mlp.pkl"
    test_model_weights(model_path)