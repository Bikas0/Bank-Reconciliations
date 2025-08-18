import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import time
import gc
from config import *

# -----------------------------
# Load model and processor once with EXTREME PERFORMANCE optimizations for RTX A5000
# -----------------------------
model_path = "./weights/DotsOCR"

# Load model with EXTREME PERFORMANCE optimizations for RTX A5000
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,  # Use float16 for maximum speed
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage
    max_memory={0: MAX_GPU_MEMORY}  # Use configurable GPU memory limit
)

# Set pad_token_id to avoid warnings
model.config.pad_token_id = model.config.eos_token_id

# Enable EXTREME PERFORMANCE model optimizations for RTX A5000
model.eval()  # Set to evaluation mode
if hasattr(model, 'half'):
    model = model.half()  # Convert to half precision

# Enable EXTREME PERFORMANCE torch optimizations for RTX A5000
torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
torch.backends.cudnn.allow_tf32 = ENABLE_TF32

# Enable extreme memory optimizations
if ENABLE_MEMORY_OPTIMIZATION:
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

# Enable Flash Attention if available
if ENABLE_FLASH_ATTENTION:
    try:
        from flash_attn import flash_attn_func
        print("✅ Flash Attention enabled for maximum speed")
    except ImportError:
        print("⚠️ Flash Attention not available, using standard attention")
        ENABLE_FLASH_ATTENTION = False

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print(f"🚀 RTX A5000 EXTREME PERFORMANCE MODE ACTIVATED!")
print(f"   Batch Size: {BATCH_SIZE} (PUSHING LIMITS)")
print(f"   Max Tokens: {MAX_TOKENS} (MAXIMUM SPEED)")
print(f"   GPU Memory: {MAX_GPU_MEMORY} (90%+ UTILIZATION)")
print(f"   Model Precision: {MODEL_PRECISION}")
print(f"   Extreme Mode: ENABLED")

# -----------------------------
# Prompt template
# -----------------------------
PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# -----------------------------
# Function to extract layout
# -----------------------------
def extract_layout_from_image(image_path: str, max_new_tokens: int = None) -> str:
    """
    Extract layout information from an image using DotsOCR model.
    EXTREME PERFORMANCE mode for maximum speed.

    Args:
        image_path (str): Path to the input image.
        max_new_tokens (int): Maximum tokens for generation (uses config if None).

    Returns:
        str: JSON string with layout information.
    """
    if max_new_tokens is None:
        max_new_tokens = MAX_TOKENS
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]

    # Prepare model input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # EXTREME PERFORMANCE inference with aggressive generation parameters
    with torch.amp.autocast('cuda'):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.pad_token_id,
            do_sample=False,  # Deterministic generation for speed
            num_beams=1,      # Single beam for maximum speed
            temperature=0.0,   # Deterministic output
            repetition_penalty=1.0,  # No repetition penalty for speed
            length_penalty=1.0,      # No length penalty for speed
            early_stopping=True,     # Stop early when possible
            use_cache=True,          # Use KV cache for speed
            # EXTREME PERFORMANCE parameters
            max_length=max_new_tokens + inputs.input_ids.shape[1],  # Exact length
            min_length=1,            # Minimum length for speed
            no_repeat_ngram_size=0,  # No n-gram blocking for speed
            bad_words_ids=None,      # No bad words filtering for speed
            force_bos_token_id=None, # No forced tokens for speed
            force_eos_token_id=None, # No forced tokens for speed
            remove_invalid_values=False,  # Skip validation for speed
            synced_gpus=False,       # Single GPU mode for speed
            # Memory optimization
            renormalize_logits=False,  # Skip renormalization for speed
            return_dict_in_generate=False,  # Return tensors for speed
        )

    # Remove input prompt from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Aggressive GPU memory cleanup for maximum performance
    if ENABLE_MEMORY_OPTIMIZATION:
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        # Force CUDA synchronization for memory cleanup
        torch.cuda.synchronize()

    return output_text[0]  # single string

def extract_layout_from_images_batch(image_paths: list, max_new_tokens: int = None, batch_size: int = None) -> dict:
    """
    Extract layout information from multiple images using EXTREME PERFORMANCE batch processing for RTX A5000.
    This processes multiple images simultaneously for MAXIMUM speed with 90%+ GPU utilization.

    Args:
        image_paths (list): List of paths to input images.
        max_new_tokens (int): Maximum tokens for generation (uses config if None).
        batch_size (int): Number of images to process simultaneously (uses config if None).

    Returns:
        dict: Dictionary with image filename as key and layout JSON as value.
    """
    # Use config values if not specified
    if max_new_tokens is None:
        max_new_tokens = MAX_TOKENS
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    results = {}
    total_images = len(image_paths)
    
    print(f"🚀 RTX A5000 EXTREME PERFORMANCE: Processing {total_images} images with batch_size={batch_size}")
    print(f"🔥 Target: 90%+ GPU utilization for MAXIMUM SPEED")
    print(f"📋 Images to process: {[os.path.basename(path) for path in image_paths]}")
    
    # Process images in EXTREME PERFORMANCE batches for RTX A5000
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"📦 EXTREME BATCH {batch_num}: {len(batch_paths)} images (GPU LOAD: TARGET 90%+)")
        print(f"   Processing: {[os.path.basename(path) for path in batch_paths]}")
        
        try:
            # Process batch of images simultaneously with EXTREME RTX A5000 optimizations
            batch_start = time.time()
            
            # GPU monitoring removed for compatibility
            
            batch_results = process_image_batch_extreme(batch_paths, max_new_tokens)
            batch_time = time.time() - batch_start
            
            # GPU monitoring removed for compatibility
            
            # Verify no duplicate processing
            for filename in batch_results:
                if filename in results:
                    print(f"⚠️  WARNING: Duplicate result for {filename} - overwriting")
            
            results.update(batch_results)
            
            # Calculate and display performance metrics
            avg_time_per_image = batch_time / len(batch_paths)
            print(f"✅ EXTREME BATCH {batch_num} completed: {len(batch_results)} images in {batch_time:.1f}s")
            print(f"⚡ Average: {avg_time_per_image:.1f}s per image (TARGET: <5s)")
            print(f"📊 Total results so far: {len(results)}/{total_images}")
            
            # Aggressive memory cleanup between batches for maximum performance
            if ENABLE_MEMORY_OPTIMIZATION:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"❌ EXTREME BATCH {batch_num} failed: {e}")
            if FALLBACK_TO_SEQUENTIAL:
                print("🔄 Falling back to individual processing...")
                
                # Fallback to individual processing - ONLY for failed batch
                for img_path in batch_paths:
                    try:
                        filename = os.path.basename(img_path)
                        # Check if already processed
                        if filename not in results:
                            layout_json = extract_layout_from_image(img_path, max_new_tokens)
                            results[filename] = layout_json
                            print(f"✅ Individual fallback: {filename}")
                        else:
                            print(f"⏭️  Skipping {filename} - already processed")
                    except Exception as e2:
                        filename = os.path.basename(img_path)
                        if filename not in results:
                            results[filename] = {"error": str(e2)}
                            print(f"❌ Individual fallback failed: {filename}")
                        else:
                            print(f"⏭️  Skipping {filename} - already processed")
            else:
                print(f"❌ Batch {batch_num} failed and fallback disabled")
                # Add error results for failed batch
                for img_path in batch_paths:
                    filename = os.path.basename(img_path)
                    if filename not in results:
                        results[filename] = {"error": f"Batch processing failed: {str(e)}"}
    
    print(f"🎉 All {total_images} images processed with EXTREME PERFORMANCE!")
    print(f"📊 Final results count: {len(results)}")
    print("✅ Status: RTX A5000 EXTREME PERFORMANCE batch OCR processing complete!")
    
    # Final verification - check for duplicates
    unique_results = len(set(results.keys()))
    if unique_results != len(results):
        print(f"⚠️  WARNING: Found {len(results) - unique_results} duplicate results!")
        # Remove duplicates, keeping the last occurrence
        seen = set()
        cleaned_results = {}
        for filename in reversed(list(results.keys())):
            if filename not in seen:
                cleaned_results[filename] = results[filename]
                seen.add(filename)
        results = cleaned_results
        print(f"🧹 Cleaned results: {len(results)} unique entries")
    
    return results

def process_image_batch_extreme(image_paths: list, max_new_tokens: int = None) -> dict:
    """
    Process a batch of images simultaneously using EXTREME PERFORMANCE optimizations.
    This is the core optimization for MAXIMUM speed with 90%+ GPU utilization.
    """
    if max_new_tokens is None:
        max_new_tokens = MAX_TOKENS
    
    batch_results = {}
    
    # Prepare batch messages
    batch_messages = []
    for img_path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
        batch_messages.append(messages)
    
    # Prepare batch inputs
    batch_texts = []
    batch_images = []
    batch_videos = []
    
    for messages in batch_messages:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        batch_texts.append(text)
        if image_inputs is not None:
            batch_images.extend(image_inputs)
        if video_inputs is not None:
            batch_videos.extend(video_inputs)
    
    # Process batch with EXTREME PERFORMANCE parameters
    inputs = processor(
        text=batch_texts,
        images=batch_images if batch_images else None,
        videos=batch_videos if batch_videos else None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    # EXTREME PERFORMANCE batch inference with aggressive parameters
    with torch.amp.autocast('cuda'):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.pad_token_id,
            do_sample=False,      # Deterministic for maximum speed
            num_beams=1,          # Single beam for maximum speed
            temperature=0.0,      # Deterministic output
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            # EXTREME PERFORMANCE parameters
            max_length=max_new_tokens + inputs.input_ids.shape[1],
            min_length=1,
            no_repeat_ngram_size=0,
            bad_words_ids=None,
            force_bos_token_id=None,
            force_eos_token_id=None,
            remove_invalid_values=False,
            synced_gpus=False,
            renormalize_logits=False,
            return_dict_in_generate=False,
        )
    
    # Process results for each image in batch
    for j, img_path in enumerate(image_paths):
        try:
            # Remove input prompt from output
            input_length = inputs.input_ids[j].shape[0]
            generated_ids_trimmed = generated_ids[j][input_length:]
            
            # Decode output
            output_text = processor.batch_decode(
                [generated_ids_trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Store result
            filename = os.path.basename(img_path)
            batch_results[filename] = output_text
            
        except Exception as e:
            filename = os.path.basename(img_path)
            batch_results[filename] = {"error": str(e)}
    
    # EXTREME PERFORMANCE memory cleanup
    if ENABLE_MEMORY_OPTIMIZATION:
        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    return batch_results
