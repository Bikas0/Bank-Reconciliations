import os
import pandas as pd


def extract_specific_columns_from_ods(UPLOAD_DIR, ods_file: str):
    """
    Extract only specific columns from ERP ODS file:
    Document No, Date, Partner, Partner Bank Acc No, Activity, Description, Debit, Credit
    Keep row if at least one of Debit or Credit is numeric.
    Drop row only if both are invalid.
    Save output CSV with same base name as input.
    """
    if not os.path.exists(ods_file):
        return None
    
    try:
        # Read the ODS file
        sheets_data = pd.read_excel(ods_file, sheet_name=None, engine='odf')
        
        for sheet_name, df in sheets_data.items():
            
            # Clean the DataFrame
            df_clean = df.fillna('')
            
            # Find the header row that contains our target columns
            header_row_index = None
            target_columns = ['DOCUMENT NO', 'DATE', 'PARTNER', 'PARTNER BANK ACC NO', 
                            'ACTIVITY', 'DESCRIPTION', 'DEBIT', 'CREDIT']
            
            for idx, row in df_clean.iterrows():
                row_values = [str(val).strip().upper() for val in row.values if str(val).strip()]
                row_text = ' '.join(row_values)
                
                # Check if this row contains our target column names
                matches = sum(1 for col in target_columns if col in row_text)
                
                if matches >= 6:  # At least 6 out of 8 columns should match
                    header_row_index = idx
                    break
            
            if header_row_index is not None:
                # Get the actual column positions
                header_row = df_clean.iloc[header_row_index]
                column_mapping = {}
                
                for col_idx, cell_value in enumerate(header_row.values):
                    cell_str = str(cell_value).strip().upper()
                    
                    if 'DOCUMENT' in cell_str and 'NO' in cell_str:
                        column_mapping['Document_No'] = col_idx
                    elif cell_str == 'DATE':
                        column_mapping['Date'] = col_idx
                    elif cell_str == 'PARTNER':
                        column_mapping['Partner'] = col_idx
                    elif 'PARTNER' in cell_str and 'BANK' in cell_str and 'ACC' in cell_str:
                        column_mapping['Partner_Bank_Acc_No'] = col_idx
                    elif cell_str == 'ACTIVITY':
                        column_mapping['Activity'] = col_idx
                    elif cell_str == 'DESCRIPTION':
                        column_mapping['Description'] = col_idx
                    elif cell_str.upper() == 'DEBIT':
                        column_mapping['Debit'] = col_idx
                    elif cell_str.upper() == 'CREDIT':
                        column_mapping['Credit'] = col_idx
                
                # Extract data starting from the row after header
                data_start_row = header_row_index + 1
                df_data = df_clean.iloc[data_start_row:].copy()
                
                # Create the final DataFrame with only the required columns
                final_data = []
                
                for idx, row in df_data.iterrows():
                    record = {}
                    
                    # Extract each required column
                    for col_name, col_idx in column_mapping.items():
                        if col_idx < len(row.values):
                            value = str(row.iloc[col_idx]).strip()
                            record[col_name] = value if value != 'nan' else ''
                        else:
                            record[col_name] = ''
                    
                    # Only add rows that have at least some data
                    if any(value and value != 'nan' for value in record.values()):
                        final_data.append(record)
                
                # Create DataFrame with the extracted data
                df_final = pd.DataFrame(final_data)
                
                # Ensure all required columns exist
                required_columns = ['Document_No', 'Date', 'Partner', 'Partner_Bank_Acc_No', 
                                  'Activity', 'Description', 'Debit', 'Credit']
                
                for col in required_columns:
                    if col not in df_final.columns:
                        df_final[col] = ''
                
                # Reorder columns
                df_final = df_final[required_columns]
                
                # Clean the data
                for col in df_final.columns:
                    df_final[col] = df_final[col].astype(str).str.strip()
                    df_final[col] = df_final[col].replace('nan', '')
                
                # Filter out completely empty rows
                df_final = df_final[df_final.apply(lambda row: any(val and val != 'nan' for val in row.values), axis=1)]
                
                # ✅ Convert Debit & Credit to numeric (invalid -> NaN)
                df_final['Debit'] = pd.to_numeric(df_final['Debit'], errors='coerce')
                df_final['Credit'] = pd.to_numeric(df_final['Credit'], errors='coerce')
                
                # ✅ Keep rows where at least one of Debit or Credit is numeric
                df_final = df_final.dropna(subset=['Debit', 'Credit'], how='all')
                
                # Keep as float (for decimals)
                df_final['Debit'] = df_final['Debit'].astype(float)
                df_final['Credit'] = df_final['Credit'].astype(float)
                
                # Build output file name based on input
                base_name = os.path.splitext(os.path.basename(ods_file))[0]
                output_file = f"{base_name}.csv"
                
                # Save to CSV
                df_final.to_csv(os.path.join(UPLOAD_DIR, output_file), index=False, encoding='utf-8')
                
                return output_file
            
            else:
                return None
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# Example usage
# extract_specific_columns_from_ods("Dataset/Pubali # 41774-ERP.ods")
