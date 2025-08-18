# -----------------------------
# Configuration File - EXTREME PERFORMANCE for RTX A5000 (24GB VRAM)
# -----------------------------
# Optimized for: 1 x RTX A5000, 16 vCPU, 62 GB RAM
# GOAL: Maximum speed with 90%+ GPU utilization

# AI Acceleration Settings - EXTREME PERFORMANCE
BATCH_SIZE = 12  # RTX A5000 can handle 12-16 images simultaneously (PUSHING LIMITS)
# - RTX A5000 (24GB VRAM): 12-16 images per batch for maximum speed
# - This will utilize 90%+ GPU load

# OCR Generation Settings - MAXIMUM SPEED
MAX_TOKENS = 1024  # Minimum tokens for maximum speed (quality vs speed trade-off)
# - 512: Ultra-fast, basic documents
# - 1024: Fast, good for most documents (RECOMMENDED for extreme speed)
# - 1536: Balanced, slightly slower

# Model Optimization Settings - EXTREME PERFORMANCE
MODEL_PRECISION = "float16"  # Use float16 for maximum speed
ENABLE_MEMORY_OPTIMIZATION = True  # Enable GPU memory optimization
MAX_GPU_MEMORY = "22GB"  # RTX A5000 has 24GB, use 22GB (90%+ utilization)

# Advanced GPU Optimizations - EXTREME MODE
ENABLE_TF32 = True  # Enable TensorFloat-32 for faster matrix operations
ENABLE_CUDNN_BENCHMARK = True  # Optimize CUDNN operations
ENABLE_FLASH_ATTENTION = True  # Use Flash Attention 2 for speed

# PDF Conversion Settings - SPEED OPTIMIZED
PDF_ZOOM = 2.0  # Reduced for speed (can be increased if quality needed)
PDF_DPI = 300   # Standard DPI for speed
OUTPUT_FORMAT = "PNG"  # Lossless quality

# Performance Tuning - EXTREME MODE
ENABLE_BATCH_PROCESSING = True  # Enable true batch processing
FALLBACK_TO_SEQUENTIAL = True   # Fallback if batch processing fails
ENABLE_FAST_GENERATION = True   # Enable fast generation parameters
ENABLE_MIXED_PRECISION = True   # Use mixed precision for speed

# Memory and CPU Optimization - MAXIMUM UTILIZATION
MAX_WORKERS = 12  # Use 12 CPU workers (75% of your 16 vCPUs)
ENABLE_CPU_OPTIMIZATION = True  # Optimize CPU operations
ENABLE_MEMORY_PINNING = True   # Pin memory for faster GPU transfer

# Output Settings
SORT_BY_FILENAME = True  # Sort pages by filename for consistent order

# JSON Output Settings
OUTPUT_DIRECTORY = "output"  # Directory to save JSON results
SAVE_JSON_FILE = True  # Whether to save results to JSON file
JSON_INDENT = 2  # JSON formatting indentation 