'''
guided sampling inference for imdb ()
same spirit as RLHF which uses the KL penalty for guidance sampling, except here I'm using guidance_scale
original RLHF paper, PPO-based optimization paper use the learned reward model directly to guide generation
this implementation is conceptually congruent but at a smaller scale and with fewer optimizations

this is the slow inference version for only imdb positive generation
matches DPO paper's first experiment (1 of 3)

mini version from FCSutopia sampling (see 193 machine)
'''

import torch
import numpy as np
import jax.numpy as jnp
from jax.nn import relu
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

class CVX_DPO_GPT2_GuidedSampling:
    def __init__(self, cvx_model_path, tokenizer_model_name, guidance_scale=1.0):
        """
        Initialize the CVX-DPO guided sampling pipeline
        
        Args:
            cvx_model_path: Path to the fine-tuned CVX-DPO model
            tokenizer_model_name: Name of the model tokenizer to use
            guidance_scale: Strength of preference guidance (higher = stronger guidance)
        """
        # Load the CVX-DPO model
        print(f"Loading CVX-DPO model from {cvx_model_path}")
        with open(cvx_model_path, 'rb') as f:
            self.cvx_model = pickle.load(f)
        
        # Extract weights
        self.theta1 = self.cvx_model.theta1
        self.theta2 = self.cvx_model.theta2
        self.guidance_scale = guidance_scale
        
        # Load GPT2 model and tokenizer
        #self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_name)
        #self.model = GPT2LMHeadModel.from_pretrained(tokenizer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(tokenizer_model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Models loaded successfully on {self.device}")
    
    def score_sequence(self, input_ids):
        """Score a sequence using the CVX-DPO model"""
        # Convert input_ids to text
        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Extract features with GPT2
        with torch.no_grad():
            # Get hidden states with exactly 60 tokens (128)
            inputs = self.tokenizer(text, return_tensors="pt", max_length=60, 
                                  truncation=True, padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.transformer(**inputs)
            hidden_states = outputs.last_hidden_state.reshape(1, -1).cpu().numpy()[0]
            
            # CVX-DPO model
            activations = relu(np.dot(hidden_states, self.theta1))
            score = np.dot(activations, self.theta2)
            
            return score
    
    def guided_generate(self, prompt, max_length=100, top_k=50, top_p=0.9, 
                        temperature=0.7, repetition_penalty=1.0):
        """
        Generate text with guidance from the CVX-DPO model
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            top_k: Number of highest probability tokens to consider for each step
            top_p: Cumulative probability threshold for nucleus sampling
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Track generated sequence
        generated = input_ids.clone()
        
        pbar = tqdm(total=max_length - input_ids.shape[1])
        
        # Generate tokens one by one with guidance
        while generated.shape[1] < max_length:
            # Get model's next token predictions
            with torch.no_grad():
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]
            
            
            next_token_logits = next_token_logits / temperature
            
            # repetition penalty
            if repetition_penalty > 1.0:
                for token_id in generated[0]:
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # convert to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # apply preference guidance for top candidates
            if self.guidance_scale > 0:
                # Get top-k token indices
                topk_values, topk_indices = torch.topk(probs, top_k, dim=-1)
                
                # For each top-k token, evaluate how it affects the preference score
                guidance_scores = []
                original_sequence = generated[0].cpu().numpy()
                
                for token_idx in topk_indices[0]:
                    # Create a candidate sequence with this token
                    candidate_sequence = np.append(original_sequence, token_idx.cpu().numpy())
                    
                    # Score the candidate sequence
                    preference_score = self.score_sequence(candidate_sequence)
                    guidance_scores.append(preference_score)
                
                # Convert to tensor and normalize
                guidance_scores = torch.tensor(guidance_scores, device=self.device)
                guidance_scores = (guidance_scores - guidance_scores.min()) / max(1e-6, guidance_scores.max() - guidance_scores.min())
                
                # Apply guidance to modify probabilities
                topk_probs = topk_values * (1 + self.guidance_scale * guidance_scores)
                
                # Re-normalize
                topk_probs = topk_probs / topk_probs.sum()
                
                # Update probabilities
                new_probs = torch.zeros_like(probs)
                new_probs[0, topk_indices[0]] = topk_probs
                probs = new_probs
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = 0
                
                # Shift the indices to the right to keep the first threshold (which should be >= p)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Mask out indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                
                # Re-normalize
                probs = probs / probs.sum()
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated
            generated = torch.cat((generated, next_token), dim=1)
            
            pbar.update(1)
            
            # Stop if we hit the end token
            if next_token[0, 0] == self.tokenizer.eos_token_id:
                break
                
        pbar.close()
        
        # Decode the generated sequence
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text

if __name__ == "__main__":
    #cvx_model_path = "/home/miria/CVXDPO/Finetuned_cvxmlp_custom_inference_ready/custom_finetuned_cvx_mlp.pkl"
    #cvx_model_path = "/home/miria/CVXDPO/Finetuned_cvxmlp_gpt2_imdb_inference_ready/gpt2_imdb_finetuned_cvx_mlp.pkl"
    #cvx_model_path = "/home/miria/CVXDPO/Finetuned_cvx_gpt2_edu_inference_ready/gpt2_attn_edu_finetuned_cvx_mlp.pkl"
    cvx_model_path = "/home/miria/CVXDPO/finetuned/Finetuned_cvx_mistral_imdb_inference_ready/mistral_imdb_finetuned_cvx_mlp.pkl"

    # Create the guided sampling pipeline
    pipeline = CVX_DPO_GPT2_GuidedSampling(
        cvx_model_path,
        tokenizer_model_name="/home/miria/CVXDPO/checkpoints/SFT_mistralai_Mistral-7B-v0.1_imdb",
        guidance_scale=2.0  # KEY: this controls guidance strength
    )
    
    # test prompts to generate positive movie reviews
    prompts = [
        "This movie was absolutely",
        "The acting in this film was so",
        "I thought the movie was"
        # "What is a sonata?",
        # "Can you tell me about geometry?",
        # "What is the point of glucose in the human body?"
    ]
    
    for prompt in prompts:
        print("\n" + "="*80)
        print(f"PROMPT: {prompt}")
        print("="*80)
        
        # Generate with guided sampling
        completion = pipeline.guided_generate(
            prompt,
            max_length=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        print(completion)