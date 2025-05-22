import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleChunkedInference:
    def __init__(self, model_name, device_map="auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            use_cache=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_chunk_size(self, context_length):
        """Simple chunk size selection based on context length"""
        if context_length <= 8000:
            return 4096
        elif context_length <= 16000:
            return 6144
        else:  # 16k-32k range
            return 8192
    
    def generate_with_chunking(self, prompt, max_new_tokens=500):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        context_length = input_ids.size(1)
        
        chunk_size = self.get_chunk_size(context_length)
        
        # If input fits in one chunk, process normally
        if context_length <= chunk_size:
            return self._generate_single(input_ids, max_new_tokens)
        
        # Otherwise, use chunked processing
        return self._generate_chunked(input_ids, chunk_size, max_new_tokens)
    
    def _generate_single(self, input_ids, max_new_tokens):
        """Generate for single chunk (no chunking needed)"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_chunked(self, input_ids, chunk_size, max_new_tokens):
        """Generate with chunked input processing"""
        context_length = input_ids.size(1)
        overlap = 256  # Small overlap to maintain context
        
        # Process all chunks to get the final hidden state
        all_logits = []
        
        for i in range(0, context_length, chunk_size - overlap):
            end_idx = min(i + chunk_size, context_length)
            chunk = input_ids[:, i:end_idx]
            
            with torch.no_grad():
                chunk_output = self.model(chunk, use_cache=False)
            
            # Keep only non-overlapping logits
            if i == 0:
                all_logits.append(chunk_output.logits)
            else:
                skip_tokens = overlap if end_idx < context_length else 0
                all_logits.append(chunk_output.logits[:, skip_tokens:])
        
        # For generation, we take the last chunk and continue from there
        last_chunk_start = max(0, context_length - chunk_size)
        last_chunk = input_ids[:, last_chunk_start:]
        
        with torch.no_grad():
            outputs = self.model.generate(
                last_chunk,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
model_name = "your-model-name"  # Replace with your model
chunked_inference = SimpleChunkedInference(model_name)

# Process your samples
def process_samples(samples, sample_range_name):
    print(f"Processing {sample_range_name} samples...")
    results = []
    
    for i, prompt in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}")
        
        # Generate response
        response = chunked_inference.generate_with_chunking(
            prompt, 
            max_new_tokens=500
        )
        results.append(response)
        
        # Optional: Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    return results

# Example usage for your three batches
# samples_4k_8k = [...]  # Your first 100 samples (4k-8k context)
# samples_8k_16k = [...] # Your second 100 samples (8k-16k context)  
# samples_16k_32k = [...] # Your third 100 samples (16k-32k context)

# results_1 = process_samples(samples_4k_8k, "4k-8k")
# results_2 = process_samples(samples_8k_16k, "8k-16k")  
# results_3 = process_samples(samples_16k_32k, "16k-32k")
