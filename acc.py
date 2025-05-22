import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
import gc

class AcceleratedStreamKVCacheInference:
    def __init__(self, model_name, max_memory_per_gpu="18GB", offload_folder=None):
        """
        Initialize with Accelerate for optimal multi-GPU distribution
        """
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16",  # Use bfloat16 mixed precision
            gradient_accumulation_steps=1,
            device_placement=True
        )
        
        print(f"Accelerator initialized on {self.accelerator.device}")
        print(f"Number of processes: {self.accelerator.num_processes}")
        print(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        # Check Flash Attention availability
        self.use_flash_attention = self._check_flash_attention()
        
        # Load model with Accelerate
        self.model, self.tokenizer = self._load_model_with_accelerate(
            model_name, max_memory_per_gpu, offload_folder
        )
        
        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check chat template support
        self.supports_chat_template = hasattr(self.tokenizer, 'chat_template') and \
                                    self.tokenizer.chat_template is not None
        
        print(f"Model loaded with Flash Attention: {self.use_flash_attention}")
        print(f"Chat template support: {self.supports_chat_template}")
        print(f"Model device map: {getattr(self.model, 'hf_device_map', 'Not available')}")
    
    def _check_flash_attention(self):
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _load_model_with_accelerate(self, model_name, max_memory_per_gpu, offload_folder):
        """Load model using Accelerate's advanced features"""
        
        # Get balanced memory distribution across available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # Convert max_memory_per_gpu to bytes if it's a string
            if isinstance(max_memory_per_gpu, str):
                memory_gb = float(max_memory_per_gpu.replace("GB", ""))
                max_memory_bytes = int(memory_gb * 1024**3)
            else:
                max_memory_bytes = max_memory_per_gpu
            
            max_memory = {i: max_memory_bytes for i in range(num_gpus)}
            device_map = "auto"
        else:
            max_memory = None
            device_map = "cpu"
        
        print(f"Loading model with max_memory: {max_memory}")
        
        # Method 1: Direct loading with device_map (simpler, recommended for most cases)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
                use_cache=True,
                trust_remote_code=True,
                offload_folder=offload_folder,
                offload_state_dict=True if offload_folder else False,
                low_cpu_mem_usage=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            print("Model loaded with direct device_map approach")
            return model, tokenizer
            
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Trying advanced Accelerate loading...")
            
            # Method 2: Advanced loading with init_empty_weights (for very large models)
            return self._load_with_empty_weights(model_name, max_memory, device_map, offload_folder)
    
    def _load_with_empty_weights(self, model_name, max_memory, device_map, offload_folder):
        """Advanced loading for very large models"""
        from accelerate import infer_auto_device_map
        
        # Initialize with empty weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # Infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer", "BloomBlock", "GPTJBlock"]
        )
        
        print(f"Inferred device map: {device_map}")
        
        # Load checkpoint and dispatch
        model = load_checkpoint_and_dispatch(
            model,
            model_name,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=True if offload_folder else False,
            dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Model loaded with empty weights approach")
        return model, tokenizer
    
    def get_chunk_size(self, context_length):
        """Determine optimal chunk size based on context length and available memory"""
        # Adjust chunk sizes based on Flash Attention and number of GPUs
        base_multiplier = 1.5 if self.use_flash_attention else 1.0
        gpu_multiplier = min(1.2, 1.0 + (self.accelerator.num_processes - 1) * 0.1)
        
        if context_length <= 8000:
            base_size = int(4096 * base_multiplier * gpu_multiplier)
        elif context_length <= 16000:
            base_size = int(6144 * base_multiplier * gpu_multiplier)
        else:  # 16k-32k range
            base_size = int(8192 * base_multiplier * gpu_multiplier)
        
        return min(base_size, 16384)  # Cap at 16k tokens per chunk
    
    def format_chat_input(self, messages, tokenize_output=False):
        """Format messages using chat template if available"""
        if self.supports_chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize_output,
                add_generation_prompt=True,
                return_tensors="pt" if tokenize_output else None
            )
        else:
            # Fallback formatting
            formatted = ""
            for msg in messages:
                if msg["role"] == "user":
                    formatted += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted += f"Assistant: {msg['content']}\n"
            formatted += "Assistant: "
            
            if tokenize_output:
                return self.tokenizer(formatted, return_tensors="pt")
            return formatted
    
    @torch.no_grad()
    def generate_with_streaming_kv_cache(self, prompt, chunk_size=None, max_new_tokens=500,
                                       do_sample=True, temperature=0.7):
        """
        Main generation method with Accelerate integration
        """
        # Handle different input types
        if isinstance(prompt, list):
            formatted_prompt = self.format_chat_input(prompt, tokenize_output=False)
        else:
            formatted_prompt = prompt
        
        # Tokenize - Accelerate will handle device placement
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids']
        
        # Move to accelerator device if needed
        if hasattr(self.accelerator, 'device'):
            input_ids = input_ids.to(self.accelerator.device)
        
        context_length = input_ids.size(1)
        
        if chunk_size is None:
            chunk_size = self.get_chunk_size(context_length)
        
        print(f"Context length: {context_length}, Chunk size: {chunk_size}")
        
        # Single chunk processing
        if context_length <= chunk_size:
            return self._generate_single_chunk(input_ids, max_new_tokens, do_sample, temperature)
        
        # Multi-chunk processing with KV cache
        past_key_values = None
        
        for i in range(0, context_length, chunk_size):
            end_idx = min(i + chunk_size, context_length)
            chunk = input_ids[:, i:end_idx]
            
            print(f"Processing chunk {i//chunk_size + 1}: tokens {i} to {end_idx-1}")
            
            if end_idx >= context_length:
                # Last chunk - generate
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
                
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Intermediate chunk - build KV cache
                outputs = self.model(chunk, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Memory management with Accelerate
            if (i // chunk_size) % 4 == 0 and i > 0:
                self.accelerator.free_memory()
                gc.collect()
    
    @torch.no_grad()
    def _generate_single_chunk(self, input_ids, max_new_tokens, do_sample, temperature):
        """Generate when input fits in single chunk"""
        print("Single chunk processing - no KV cache chunking needed")
        
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
        """Generate with benchmark-specific settings"""
        benchmark_settings = {
            "multiple_choice": {"do_sample": False, "temperature": None, "max_new_tokens": 1},
            "qa": {"do_sample": False, "temperature": None, "max_new_tokens": 100},
            "chat": {"do_sample": True, "temperature": 0.7, "max_new_tokens": 500},
            "summarization": {"do_sample": True, "temperature": 0.3, "max_new_tokens": 200}
        }
        
        settings = benchmark_settings.get(benchmark_type, benchmark_settings["chat"])
        return self.generate_with_streaming_kv_cache(prompt, **settings)
    
    def batch_process_samples(self, samples, sample_type="chat", chunk_size=None):
        """Process multiple samples with Accelerate optimizations"""
        print(f"Processing {len(samples)} samples of type: {sample_type}")
        results = []
        
        for i, prompt in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            
            try:
                result = self.benchmark_generate(prompt, benchmark_type=sample_type)
                results.append({
                    "index": i,
                    "prompt_length": len(prompt) if isinstance(prompt, str) else len(str(prompt)),
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {str(e)}")
                results.append({
                    "index": i,
                    "prompt_length": len(prompt) if isinstance(prompt, str) else len(str(prompt)),
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
            
            # Enhanced memory cleanup with Accelerate
            if (i + 1) % 10 == 0:
                self.accelerator.free_memory()
                gc.collect()
                print(f"Memory cleanup after sample {i+1}")
        
        return results
    
    def get_memory_usage(self):
        """Get current memory usage across all devices"""
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_info[f"GPU_{i}"] = {
                    "allocated": f"{allocated:.1f}GB",
                    "reserved": f"{reserved:.1f}GB"
                }
        
        return memory_info

# Usage Example
def main():
    # Initialize with Accelerate
    model_name = "your-model-name"
    inference = AcceleratedStreamKVCacheInference(
        model_name=model_name,
        max_memory_per_gpu="18GB",
        offload_folder="./offload"  # Optional: offload to disk if needed
    )
    
    # Check initial memory usage
    print("Initial memory usage:")
    memory_info = inference.get_memory_usage()
    for gpu, info in memory_info.items():
        print(f"{gpu}: {info}")
    
    # Test samples
    test_samples = [
        "Your test prompt here...",
        [{"role": "user", "content": "Hello, how are you?"}],  # Message format
        # ... more samples
    ]
    
    # Process samples
    results = inference.batch_process_samples(
        test_samples,
        sample_type="chat"
    )
    
    # Print results summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {successful}/{len(results)} samples successful")
    
    # Final memory usage
    print("\nFinal memory usage:")
    memory_info = inference.get_memory_usage()
    for gpu, info in memory_info.items():
        print(f"{gpu}: {info}")

if __name__ == "__main__":
    main()
