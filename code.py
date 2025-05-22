import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

class StreamKVCacheInference:
    def __init__(self, model_name, device_map="auto", max_memory_per_gpu="10GB"):
        """
        Initialize the model with optimizations for long context inference
        """
        # Check Flash Attention availability
        self.use_flash_attention = self._check_flash_attention()
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            max_memory={i: max_memory_per_gpu for i in range(8)},
            attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
            use_cache=True,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded with Flash Attention: {self.use_flash_attention}")
        print(f"Model dtype: {self.model.dtype}")
        print(f"Device map: {self.model.hf_device_map}")
    
    def _check_flash_attention(self):
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def get_chunk_size(self, context_length):
        """Determine optimal chunk size based on context length"""
        if context_length <= 8000:
            return 6144 if self.use_flash_attention else 4096
        elif context_length <= 16000:
            return 8192 if self.use_flash_attention else 6144
        else:  # 16k-32k range
            return 12288 if self.use_flash_attention else 8192
    
    def generate_with_streaming_kv_cache(self, prompt, chunk_size=None, max_new_tokens=500, 
                                       do_sample=True, temperature=0.7):
        """
        Main generation method with streaming KV cache processing
        Single pass through input - builds cache and generates in one loop
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids']
        context_length = input_ids.size(1)
        
        # Determine chunk size if not provided
        if chunk_size is None:
            chunk_size = self.get_chunk_size(context_length)
        
        print(f"Context length: {context_length}, Chunk size: {chunk_size}")
        
        # If input fits in one chunk, process normally
        if context_length <= chunk_size:
            return self._generate_single_chunk(input_ids, max_new_tokens, do_sample, temperature)
        
        # Stream processing with KV cache
        past_key_values = None
        
        for i in range(0, context_length, chunk_size):
            end_idx = min(i + chunk_size, context_length)
            chunk = input_ids[:, i:end_idx]
            
            print(f"Processing chunk {i//chunk_size + 1}: tokens {i} to {end_idx-1}")
            
            with torch.no_grad():
                if end_idx >= context_length:
                    # Last chunk - generate directly and return
                    print("Last chunk - generating response...")
                    outputs = self.model.generate(
                        chunk,
                        past_key_values=past_key_values,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode and return result
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return generated_text
                    
                else:
                    # Intermediate chunk - just build KV cache
                    outputs = self.model(
                        chunk, 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
            
            # Memory management - clear cache every few chunks
            if (i // chunk_size) % 4 == 0 and i > 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    def _generate_single_chunk(self, input_ids, max_new_tokens, do_sample, temperature):
        """Generate when input fits in single chunk"""
        print("Single chunk processing - no KV cache chunking needed")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def benchmark_generate(self, prompt, benchmark_type="chat"):
        """
        Generate with benchmark-specific settings
        """
        benchmark_settings = {
            "multiple_choice": {
                "do_sample": False,
                "temperature": None,
                "max_new_tokens": 1
            },
            "qa": {
                "do_sample": False, 
                "temperature": None,
                "max_new_tokens": 100
            },
            "chat": {
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 500
            },
            "summarization": {
                "do_sample": True,
                "temperature": 0.3,
                "max_new_tokens": 200
            }
        }
        
        settings = benchmark_settings.get(benchmark_type, benchmark_settings["chat"])
        
        return self.generate_with_streaming_kv_cache(
            prompt,
            **settings
        )
    
    def batch_process_samples(self, samples, sample_type="chat", chunk_size=None):
        """
        Process multiple samples efficiently
        """
        print(f"Processing {len(samples)} samples of type: {sample_type}")
        results = []
        
        for i, prompt in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            
            try:
                result = self.benchmark_generate(prompt, benchmark_type=sample_type)
                results.append({
                    "index": i,
                    "prompt_length": len(prompt),
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {str(e)}")
                results.append({
                    "index": i,
                    "prompt_length": len(prompt),
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
            
            # Memory cleanup every 10 samples
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Memory cleanup after sample {i+1}")
        
        return results

# Usage Example
def main():
    # Initialize model
    model_name = "your-model-name"  # Replace with your actual model
    inference = StreamKVCacheInference(
        model_name=model_name,
        device_map="auto",
        max_memory_per_gpu="10GB"
    )
    
    # Example samples for your three batches
    samples_4k_8k = [
        "Your 4k-8k context samples here...",
        # ... more samples
    ]
    
    samples_8k_16k = [
        "Your 8k-16k context samples here...",
        # ... more samples  
    ]
    
    samples_16k_32k = [
        "Your 16k-32k context samples here...",
        # ... more samples
    ]
    
    # Process each batch
    print("=== Processing 4k-8k samples ===")
    results_1 = inference.batch_process_samples(
        samples_4k_8k, 
        sample_type="chat",
        chunk_size=6144
    )
    
    print("\n=== Processing 8k-16k samples ===")  
    results_2 = inference.batch_process_samples(
        samples_8k_16k,
        sample_type="chat", 
        chunk_size=8192
    )
    
    print("\n=== Processing 16k-32k samples ===")
    results_3 = inference.batch_process_samples(
        samples_16k_32k,
        sample_type="chat",
        chunk_size=12288
    )
    
    # Print summary
    all_results = [results_1, results_2, results_3]
    batch_names = ["4k-8k", "8k-16k", "16k-32k"]
    
    for batch_results, batch_name in zip(all_results, batch_names):
        successful = sum(1 for r in batch_results if r["success"])
        print(f"\n{batch_name} batch: {successful}/{len(batch_results)} successful")

# For single sample testing
def test_single_sample():
    model_name = "your-model-name"
    inference = StreamKVCacheInference(model_name)
    
    # Test with a long prompt
    long_prompt = "Your very long test prompt here..." * 1000  # Make it long
    
    result = inference.generate_with_streaming_kv_cache(
        prompt=long_prompt,
        chunk_size=8192,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )
    
    print("Generated response:")
    print(result)

if __name__ == "__main__":
    main()
    # Or run: test_single_sample()
