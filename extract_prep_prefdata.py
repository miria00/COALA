'''
gets preference data into shape, ready for extract.py
get edu and ultra binary data to match imdb data shape
ready for extract.py
'''

import os
import json

def process_dataset(input_file, output_dir):
    """
    Process the JSON dataset and create txt files for positive and negative examples.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to store the output files
    """
    # Create output directories
    pos_dir = os.path.join(output_dir, "pos")
    neg_dir = os.path.join(output_dir, "neg")
    
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    
    # Read and process the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                # Extract prompt, chosen, and rejected responses
                prompt = data.get("prompt", "")
                chosen = data.get("chosen", "")
                rejected = data.get("rejected", "")
                
                # Skip empty entries
                if not prompt or not chosen or not rejected:
                    continue
                
                # Create positive example (prompt + chosen)
                pos_text = prompt + chosen
                pos_file = os.path.join(pos_dir, f"example_{i:06d}.txt")
                
                # Create negative example (prompt + rejected)
                neg_text = prompt + rejected
                neg_file = os.path.join(neg_dir, f"example_{i:06d}.txt")
                
                # Write to files
                with open(pos_file, 'w', encoding='utf-8') as pos_f:
                    pos_f.write(pos_text)
                
                with open(neg_file, 'w', encoding='utf-8') as neg_f:
                    neg_f.write(neg_text)
                
                if i % 100 == 0:
                    print(f"Processed {i} examples")
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i}. Skipping.")
            except Exception as e:
                print(f"Error processing line {i}: {e}")
    
    # Count files in each directory
    pos_count = len(os.listdir(pos_dir))
    neg_count = len(os.listdir(neg_dir))
    
    print(f"\nProcessing complete!")
    print(f"Positive examples: {pos_count}")
    print(f"Negative examples: {neg_count}")

if __name__ == "__main__":
    #input_file = "/home/miria/CVXDPO/train_test_edu_dataset_full.json"
    input_file = "/home/miria/CVXDPO/train_test_ultra_dataset_full.json"
    output_dir = "/home/miria/CVXDPO/datasets/ultra_"
    os.makedirs(output_dir, exist_ok=True)

    
    print(f"Processing dataset from {input_file} to {output_dir}...")
    process_dataset(input_file, output_dir)