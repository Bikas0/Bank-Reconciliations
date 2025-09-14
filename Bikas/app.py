import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from bankReconciliationAgent import workflow, BankReconciliationState   # import both workflow + state
from erp_ods_to_csv import extract_specific_columns_from_ods

app = FastAPI()

# -------------------- Middleware --------------------
# Enable CORS (so frontend clients can access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all headers
)

UPLOAD_DIR = "Datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/reconcile")
async def reconcile(bank_file: UploadFile = File(...), erp_file: UploadFile = File(...)):
    """
    Upload Bank CSV + ERP ODS → process → reconciliation result
    """
    try:
        # --- Save uploaded files ---
        bank_path = os.path.join(UPLOAD_DIR, bank_file.filename)
        erp_path = os.path.join(UPLOAD_DIR, erp_file.filename)

        with open(bank_path, "wb") as f:
            shutil.copyfileobj(bank_file.file, f)
        with open(erp_path, "wb") as f:
            shutil.copyfileobj(erp_file.file, f)

        # --- Step 1: Convert ERP ODS -> CSV ---
        erp_csv = extract_specific_columns_from_ods(UPLOAD_DIR, erp_path)
        if not erp_csv:
            return [
                {
                    "Status": False,
                    "Message": "Failed to extract ERP ODS file",
                    "Data": {}
                }
            ]

        erp_csv_path = os.path.join(UPLOAD_DIR, erp_csv)

        # --- Step 2: Run Agent workflow ---
        initial_state = {
            "erp_data": erp_csv_path,
            "bank_data": bank_path
        }
        raw_result = workflow.invoke(initial_state, {"recursion_limit": 100})

        # Wrap into BankReconciliationState so you can use dot-access
        result = BankReconciliationState(**raw_result)

        # --- Step 3: Return result in required format ---
        return [
            {
                "Status": True,
                "Message": "Reconciliation completed successfully",
                "Data": {
                    "unadjusted_erp_balance": result.unadjusted_erp_balance,
                    "unadjusted_bank_balance": result.unadjusted_bank_balance,
                    "adjusted_bank_balance": result.adjusted_bank_balance,
                    "adjusted_erp_balance": result.adjusted_erp_balance,
                    "amount_difference": result.amount_difference,
                    "response": result.response_messages
                }
            }
        ]

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [
            {
                "Status": False,
                "Message": "Reconciliation failed",
                "Data": {
                    "error": str(e)
                }
            }
        ]


if __name__ == "__main__":
    # Enable hot-reload during development
    uvicorn.run(
        "app:app",        # use app:app import path
        host="0.0.0.0",
        port=8000,
        reload=True       # 👈 auto-reload on code changes
    )
