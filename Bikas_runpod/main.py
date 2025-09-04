import os
import shutil
import tempfile
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from pdf_to_images import convert_pdf_to_images
from hug_infer import extract_layout_from_image, extract_layout_from_images_batch
from config import *
from table import process_bank_statement_json_to_csv

def process_images_sequentially(image_paths):
    """Process images one by one (fallback method)"""
    results = {}
    for img_path in image_paths:
        try:
            layout_json = extract_layout_from_image(img_path)
            results[os.path.basename(img_path)] = layout_json
            print(f"✅ Sequential: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"❌ Page {os.path.basename(img_path)} Error: {e}")
            results[os.path.basename(img_path)] = {"error": str(e)}
    return results


def delete_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Successfully deleted folder and its contents: {folder_path}")
        else:
            print(f"Folder does not exist: {folder_path}")
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {e}")


app = FastAPI(title="PDF → Image → OCR API 🚀")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, convert it to images, extract layout via DotsOCR using batch processing,
    and return JSON results for all pages.
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    start_time = datetime.now()
    
    print(f"🚀 [REQUEST-{request_id}] Starting new PDF processing request")
    print(f"📁 [REQUEST-{request_id}] File: {file.filename}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        # Save uploaded PDF
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"💾 [REQUEST-{request_id}] PDF saved to temporary directory")

        # Convert PDF → Images
        output_folder = os.path.join(tmpdir, "images")
        print(f"🔄 [REQUEST-{request_id}] Converting PDF to images...")
        
        image_paths = convert_pdf_to_images(
            pdf_path, 
            output_folder=output_folder, 
            method="pymupdf",
            zoom=PDF_ZOOM,
            format=OUTPUT_FORMAT
        )
        if not image_paths:
            return JSONResponse({"error": "PDF conversion failed"}, status_code=500)

        # Sort images by filename if enabled
        if SORT_BY_FILENAME:
            print(f"🔄 [REQUEST-{request_id}] Sorting images by filename...")
            image_paths = sorted(image_paths, key=lambda x: os.path.basename(x))
            print(f"📋 [REQUEST-{request_id}] Sorted image paths: {[os.path.basename(path) for path in image_paths]}")

        print(f"✅ [REQUEST-{request_id}] PDF converted to {len(image_paths)} images")
        print(f"🚀 [REQUEST-{request_id}] Starting batch OCR processing...")
        print(f"📋 [REQUEST-{request_id}] Image paths: {[os.path.basename(path) for path in image_paths]}")

        # Extract layout for all images using RTX A5000 EXTREME PERFORMANCE batch processing
        if ENABLE_BATCH_PROCESSING:
            try:
                # Use EXTREME PERFORMANCE batch processing for RTX A5000 (24GB VRAM)
                # This processes multiple images simultaneously using AI acceleration with 90%+ GPU utilization
                print(f"⏳ [REQUEST-{request_id}] Status: Processing OCR with RTX A5000 EXTREME PERFORMANCE mode...")
                print(f"🚀 [REQUEST-{request_id}] RTX A5000 EXTREME PERFORMANCE Configuration:")
                print(f"   Batch Size: {BATCH_SIZE} (PUSHING LIMITS for 90%+ GPU load)")
                print(f"   Max Tokens: {MAX_TOKENS} (MAXIMUM SPEED)")
                print(f"   GPU Memory: {MAX_GPU_MEMORY} (90%+ VRAM utilization)")
                print(f"   Model Precision: {MODEL_PRECISION}")
                print(f"   Extreme Mode: ACTIVATED")
                
                # GPU monitoring removed for compatibility
                
                print(f"🔍 [REQUEST-{request_id}] Starting batch processing for {len(image_paths)} images...")
                results = extract_layout_from_images_batch(image_paths, batch_size=BATCH_SIZE, max_new_tokens=MAX_TOKENS)
                
                print(f"🔍 [REQUEST-{request_id}] Batch processing completed. Results count: {len(results)}")
                
                print(f"✅ [REQUEST-{request_id}] RTX A5000 EXTREME PERFORMANCE OCR completed for {len(results)} pages")
                print(f"✅ [REQUEST-{request_id}] Status: RTX A5000 EXTREME PERFORMANCE processing complete!")
                
            except Exception as e:
                print(f"❌ [REQUEST-{request_id}] RTX A5000 EXTREME PERFORMANCE failed: {e}")
                if FALLBACK_TO_SEQUENTIAL:
                    print(f"🔄 [REQUEST-{request_id}] Falling back to sequential processing...")
                    print(f"⏳ [REQUEST-{request_id}] Status: Fallback processing...")
                    results = process_images_sequentially(image_paths)
                    print(f"✅ [REQUEST-{request_id}] Status: Fallback processing complete!")
                else:
                    return JSONResponse({"error": f"RTX A5000 EXTREME PERFORMANCE failed: {str(e)}"}, status_code=500)
        else:
            # Use sequential processing
            print(f"⏳ [REQUEST-{request_id}] Status: Sequential processing...")
            results = process_images_sequentially(image_paths)
            print(f"✅ [REQUEST-{request_id}] Status: Sequential processing complete!")

        # Final completion summary
        print(f"🎯 [REQUEST-{request_id}] All processing completed successfully!")
        print(f"📊 [REQUEST-{request_id}] Final results count: {len(results)}")
        print(f"📋 [REQUEST-{request_id}] Final result keys: {list(results.keys())}")
        print(f"📊 [REQUEST-{request_id}] Final Status: DONE")

        # Return results as JSON
        completion_time = datetime.now()
        processing_duration = (completion_time - start_time).total_seconds()
        
        response_data = {
            "pdf": file.filename, 
            "total_pages": len(results), 
            "results": results,
            "status": "done",
            "started_at": start_time.isoformat(),
            "completed_at": completion_time.isoformat(),
            "processing_duration_seconds": round(processing_duration, 2),
            "message": f"Successfully processed {len(results)} pages from {file.filename} in {round(processing_duration, 2)} seconds"
        }
        
        # Save results to JSON file
        if SAVE_JSON_FILE:
            try:
                delete_folder(OUTPUT_DIRECTORY)
                # Create output directory if it doesn't exist
                os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
                
                # Generate filename based on PDF name
                pdf_name = os.path.splitext(file.filename)[0]  # Remove .pdf extension
                json_filename = f"{pdf_name}.json"
                json_path = os.path.join(OUTPUT_DIRECTORY, json_filename)
                
                # Save JSON file
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(response_data, json_file, indent=JSON_INDENT, ensure_ascii=False)
                
                print(f"💾 [REQUEST-{request_id}] JSON results saved to: {json_path}")
                process_bank_statement_json_to_csv(OUTPUT_DIRECTORY)
                # Add file path to response
                response_data["json_file_path"] = json_path
                
            except Exception as e:
                print(f"⚠️ [REQUEST-{request_id}] Warning: Could not save JSON file: {e}")
                response_data["json_file_path"] = "Failed to save"
        else:
            response_data["json_file_path"] = "JSON saving disabled"
        
        print(f"🎯 [REQUEST-{request_id}] Returning response with {len(results)} results")
        return response_data


@app.get("/api")
def root():
    return {"message": "PDF → Image → OCR API is running 🚀"}


# ------------------------------
# Production-ready entry point
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)