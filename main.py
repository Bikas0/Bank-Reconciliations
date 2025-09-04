import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel, Field
from datetime import date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# State definition
class BankReconciliationState(BaseModel):
    bank_data: str  # file path for bank data
    erp_data: str   # file path for ERP data
    bank_columns: Optional[dict] = None
    erp_columns: Optional[dict] = None

from pydantic import BaseModel, Field

class ColumnSchema(BaseModel):
    transaction_date: str = Field(..., description="Column name for transaction date")
    debit: str = Field(..., description="Column name for debit amount")
    credit: str = Field(..., description="Column name for credit amount")


structured_model_column = llm.with_structured_output(ColumnSchema)


# Reusable function for detecting columns
def detect_columns(file_path: str):
    df = pd.read_csv(file_path)
    subset_data = df.head(5)
    prompt = f"""
    From the following dataset, identify which column corresponds to:
    - transaction_date
    - debit
    - credit

    Dataset sample:
    {subset_data.to_dict(orient='records')}
    """
    return structured_model_column.invoke(prompt)


# Node: find columns in bank dataset
def find_bank_columns(state: BankReconciliationState):
    result = detect_columns(state.bank_data)
    return {"bank_columns": result}


# Node: find columns in ERP dataset
def find_erp_columns(state: BankReconciliationState):
    result = detect_columns(state.erp_data)
    return {"erp_columns": result}

# Build workflow
graph = StateGraph(BankReconciliationState)

graph.add_node("find_bank_columns", find_bank_columns)
graph.add_node("find_erp_columns", find_erp_columns)

# Run them in parallel (both start from START)
graph.add_edge(START, "find_bank_columns")
graph.add_edge(START, "find_erp_columns")

# Both connect to END
graph.add_edge("find_bank_columns", END)
graph.add_edge("find_erp_columns", END)

# Compile workflow
workflow = graph.compile()
# Run with both datasets
initial_state = {
    "erp_data": "Dataset/Pubali # 41774-ERP.csv",
    "bank_data": "Dataset/Pubali # 41774.csv"
}
result = workflow.invoke(initial_state)

print("Bank Columns:", result["bank_columns"])
print("ERP Columns:", result["erp_columns"])