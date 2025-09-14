import os
import re
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_groq import ChatGroq
from datetime import date, datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import Graph, StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure LLM (adjust model name as needed)
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

class ColumnSchema(BaseModel):
    transaction_date: str = Field(..., description="Column name for transaction date")
    debit: str = Field(..., description="Column name for debit amount")
    credit: str = Field(..., description="Column name for credit amount")


class BankReconciliationState(BaseModel):
    bank_data: str
    erp_data: str
    bank_columns: Optional[ColumnSchema] = None
    erp_columns: Optional[ColumnSchema] = None
    unadjusted_erp_balance: Optional[float] = 0
    unadjusted_bank_balance: Optional[float] = 0
    adjusted_erp_balance: Optional[float] = 0
    adjusted_bank_balance: Optional[float] = 0
    messages: Optional[str] = None
    amount_difference: Optional[int] = 0
    response_messages: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True
    }

# Create structured output wrapper if desired (keeps original name)
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
    # return as dict (langgraph expects dict)
    # print(result)
    return {"bank_columns": result}

# Node: find columns in ERP dataset
def find_erp_columns(state: BankReconciliationState):
    result = detect_columns(state.erp_data)
    # print(result)
    return {"erp_columns": result}


def reconciliations(state: BankReconciliationState):
    """LLM-powered reconciliation function that generates Python code for calculations.
    Always return a dict of state updates (not the model object or END constant).
    """

    bank_df = pd.read_csv(state.bank_data)
    erp_df = pd.read_csv(state.erp_data)
    
    # Extract dynamic column names from state
    bank_transaction_date = state.bank_columns.transaction_date
    bank_credit = state.bank_columns.credit
    bank_debit = state.bank_columns.debit
    erp_transaction_date = state.erp_columns.transaction_date
    erp_credit = state.erp_columns.credit
    erp_debit = state.erp_columns.debit

    # Build prompt for LLM
    bank_sample = bank_df.head(5).to_string() if not bank_df.empty else "No data"
    erp_sample = erp_df.head(5).to_string() if not erp_df.empty else "No data"

    prompt = f"""You are a financial analyst. Write Python code to calculate net balance for two datasets.

        Bank dataset columns:
        - Transaction date: "{bank_transaction_date}"
        - Debit: "{bank_debit}" 
        - Credit: "{bank_credit}"

        ERP dataset columns:
        - Transaction date: "{erp_transaction_date}"
        - Debit: "{erp_debit}"
        - Credit: "{erp_credit}"

        Sample bank data:
        {bank_sample}

        Sample ERP data:
        {erp_sample}

        Task: Write Python code to calculate net balance for bank dataset is (total_credit - total_debit) and net balance for erp dataset is (total_debit - total_credit).

        Requirements:
        1. Convert columns to numeric, handling commas and NaN values as 0
        2. Calculate: bank_net_balance = bank_total_credit - bank_total_debit
        3. Calculate: erp_net_balance = erp_total_debit - erp_total_credit 
        4. Use the exact column names provided above
        5. Return only executable Python code, no explanations or markdown

        Example format:
        # Clean bank data
        bank_df['{bank_credit}'] = pd.to_numeric(bank_df['{bank_credit}'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        bank_df['{bank_debit}'] = pd.to_numeric(bank_df['{bank_debit}'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        bank_total_credit = bank_df['{bank_credit}'].sum()
        bank_total_debit = bank_df['{bank_debit}'].sum()
        bank_net_balance = bank_total_credit - bank_total_debit

        # Clean ERP data
        erp_df['{erp_credit}'] = pd.to_numeric(erp_df['{erp_credit}'], errors='coerce').fillna(0)
        erp_df['{erp_debit}'] = pd.to_numeric(erp_df['{erp_debit}'], errors='coerce').fillna(0)
        erp_total_credit = erp_df['{erp_credit}'].sum()
        erp_total_debit = erp_df['{erp_debit}'].sum()
        erp_net_balance = erp_total_debit - erp_total_credit"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'\n?```', '', code).strip()

        # Execute code in safe local scope
        local_vars = {"bank_df": bank_df.copy(), "erp_df": erp_df.copy(), "pd": pd}
        exec(code, {"pd": pd}, local_vars)

        bank_net_balance = local_vars.get("bank_net_balance")
        erp_net_balance = local_vars.get("erp_net_balance")

        if bank_net_balance is None or erp_net_balance is None:
            raise ValueError("LLM code did not produce required variables")
        # print(f"💰 LLM Calculated - Bank total: {bank_net_balance}, ERP total: {erp_net_balance}")
        # Update state fields via dict
        return {
            "unadjusted_bank_balance": float(bank_net_balance),
            "unadjusted_erp_balance": float(erp_net_balance)
        }

    except Exception as e:
        # Fallback hardcoded calculation
        bank_df = bank_df.copy()
        erp_df = erp_df.copy()

        # Safe column conversions (use provided names)
        try:
            bank_df[bank_credit] = pd.to_numeric(bank_df[bank_credit].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)
            bank_df[bank_debit] = pd.to_numeric(bank_df[bank_debit].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)
            bank_total_credit = bank_df[bank_credit].sum()
            bank_total_debit = bank_df[bank_debit].sum()
            bank_net_balance = bank_total_credit - bank_total_debit
        except Exception:
            bank_net_balance = 0.0

        try:
            erp_df[erp_credit] = pd.to_numeric(erp_df[erp_credit], errors='coerce').fillna(0)
            erp_df[erp_debit] = pd.to_numeric(erp_df[erp_debit], errors='coerce').fillna(0)
            erp_total_credit = erp_df[erp_credit].sum()
            erp_total_debit = erp_df[erp_debit].sum()
            erp_net_balance = erp_total_debit - erp_total_credit
        except Exception:
            erp_net_balance = 0.0
        # print(f"💰 Fallback Calculated - Bank total: {bank_net_balance}, ERP total: {erp_net_balance}")
    
        return {
            "unadjusted_bank_balance": float(bank_net_balance),
            "unadjusted_erp_balance": float(erp_net_balance)
        }

# print("✅ Helper functions defined")

def compare_unadjusted_balances(state: BankReconciliationState):
    """
    Compare the latest Bank and ERP chunk balances.
    Always return a dict of updates.
    """
    try:
        bank_val = float(state.unadjusted_bank_balance or 0)
        erp_val = float(state.unadjusted_erp_balance or 0)
        # print(bank_val, erp_val)
        if bank_val == erp_val:
            # print("Bank_total_amount: ", float((state.unadjusted_bank_balance or 0) + bank_val),"\n", "Erp_total_amount: ", float((state.unadjusted_erp_balance or 0) + erp_val))
            # add to totals and advance
            return {
                "adjusted_bank_balance": bank_val,
                "adjusted_erp_balance": erp_val,
                "messages": "compare_total_balance"
            }
        else:
            # print("======================= Call Eliminate")
            return {"messages": "eliminate_and_subtrct_values"}
    except Exception as e:
        return {"error": str(e)}
    
# Clean data helper
def clean_dataframe(df, debit_col, credit_col):
    df = df.copy()
    df[debit_col] = df[debit_col].fillna(0)
    df[credit_col] = df[credit_col].fillna(0)

    def clean_numeric(series):
        return (
            series.astype(str)
            .str.replace(",", "", regex=True)
            .str.replace(" ", "", regex=True)
            .str.strip()
            .replace("", "0")
            .astype(float)
        )

    df[debit_col] = clean_numeric(df[debit_col])
    df[credit_col] = clean_numeric(df[credit_col])
    return df

def normalize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def eliminate_and_subtrct_values(state: BankReconciliationState):
    bank_df = pd.read_csv(state.bank_data)
    erp_df = pd.read_csv(state.erp_data)

    bank_credit = state.bank_columns.credit
    bank_debit = state.bank_columns.debit
    erp_credit = state.erp_columns.credit
    erp_debit = state.erp_columns.debit

    if bank_df is None or erp_df is None:
        return {"messages": "compare_total_balance"}

    bank_df = clean_dataframe(bank_df, bank_debit, bank_credit)
    erp_df = clean_dataframe(erp_df, erp_debit, erp_credit)

    # --- Reconciliation (one-to-one match removal) ---
    bank_remaining = bank_df.copy()
    erp_remaining = erp_df.copy()

    matched_bank_idx = []
    matched_erp_idx = []
    for i, b_row in bank_remaining.iterrows():
        for j, e_row in erp_remaining.iterrows():
            if (b_row[bank_debit] == e_row[erp_credit]) and (b_row[bank_credit] == e_row[erp_debit]):
                matched_bank_idx.append(i)
                matched_erp_idx.append(j)
                # remove this erp row so it can’t be matched again
                erp_remaining = erp_remaining.drop(j)
                break

    # Remove matched bank rows (ERP already updated inside loop)
    bank_remaining = bank_remaining.drop(matched_bank_idx)
    erp_remaining = erp_df.drop(matched_erp_idx)
    # print(bank_remaining, "\n\n", erp_remaining, "\n\n")
    # print("✅ Updated balances:\nERP:", state.unadjusted_erp_balance, "\nBank:", state.unadjusted_bank_balance)
    unadjusted_erp_balances = (
        float(state.unadjusted_erp_balance or 0)
        - (bank_remaining[bank_debit].sum() if not bank_remaining.empty else 0)
        + (bank_remaining[bank_credit].sum() if not bank_remaining.empty else 0)
    )

    unadjusted_bank_balances = (
        float(state.unadjusted_bank_balance or 0)
        - (erp_remaining[erp_credit].sum() if not erp_remaining.empty else 0)
        + (erp_remaining[erp_debit].sum() if not erp_remaining.empty else 0)
    )

    print("Adjusted ERP: ", unadjusted_erp_balances, "\n", "Adjusted ERP: ", unadjusted_bank_balances)

    # Return updates as dict
    return {
        "adjusted_bank_balance": float((state.adjusted_bank_balance or 0) + unadjusted_bank_balances),
        "adjusted_erp_balance": float((state.adjusted_erp_balance or 0) + unadjusted_erp_balances),
    }


def compare_total_balance(state: BankReconciliationState):
    if state.adjusted_bank_balance > state.adjusted_erp_balance:
        state.amount_difference = state.adjusted_bank_balance - state.adjusted_erp_balance
        state.response_messages = f"The bank balance of {state.adjusted_bank_balance} is greater than the ERP balance of {state.adjusted_erp_balance}, with a difference of {state.amount_difference}."
        
    elif state.adjusted_bank_balance < state.adjusted_erp_balance:
        state.amount_difference = state.adjusted_erp_balance - state.adjusted_bank_balance
        state.response_messages =  "The ERP balance of {state.adjusted_erp_balance} is greater than the bank balance of {state.adjusted_bank_balance}, with a difference of {state.amount_difference}."

    else:
        state.amount_difference = 0
        state.response_messages = f"The ERP balance is {state.adjusted_erp_balance} and the bank balance is {state.adjusted_bank_balance}; both are equal."
    
    print(state.response_messages)

    return state

# Conditional edges
def route_after_comparison(state: BankReconciliationState):
    messages = state.messages or ""
    if isinstance(messages, str) and "eliminate_and_subtrct_values" in messages:
        return "eliminate_and_subtrct_values"
    else:
        return "compare_unadjusted_balances"
    
# --- Build workflow ---
graph = StateGraph(BankReconciliationState)

# Add nodes
graph.add_node("find_bank_columns", find_bank_columns)
graph.add_node("find_erp_columns", find_erp_columns)
graph.add_node("reconciliations", reconciliations)
graph.add_node("compare_unadjusted_balances", compare_unadjusted_balances)
graph.add_node("eliminate_and_subtrct_values", eliminate_and_subtrct_values)
graph.add_node("compare_total_balance", compare_total_balance)

# Connect edges
graph.add_edge(START, "find_bank_columns")
graph.add_edge(START, "find_erp_columns")
graph.add_edge("find_erp_columns", "reconciliations")
graph.add_edge("find_bank_columns", "reconciliations")
graph.add_edge("reconciliations", "compare_unadjusted_balances")

graph.add_conditional_edges(
    "compare_unadjusted_balances",
    route_after_comparison,
    {
        "eliminate_and_subtrct_values": "eliminate_and_subtrct_values",
        "compare_total_balance": "compare_total_balance"
    }
)
graph.add_edge("eliminate_and_subtrct_values", "compare_total_balance")
graph.add_edge("compare_total_balance", END)

# Compile workflow
workflow = graph.compile()
# print("✅ Workflow built and compiled")

# initial_state = {
#     "erp_data": "Dataset/Pubali # 41774-ERP.csv",
#     "bank_data": "Dataset/Pubali # 41774.csv"
# }

# try:
#     result = workflow.invoke(initial_state, {"recursion_limit": 100})
#     print("✅ Workflow executed successfully")
#     # print(result)
# except Exception as e:
#     print(f"❌ Error running workflow: {e}")
#     import traceback
#     traceback.print_exc()
