
# import json
# from bs4 import BeautifulSoup
# import csv
# from io import StringIO
# import os
# import glob

# def process_bank_statement_json_to_csv(input_directory):
#     """
#     Process a bank statement JSON file in the input directory and extract table data to a CSV file
#     in the same directory with the same name but .csv extension.
    
#     Args:
#         input_directory (str): Path to the directory containing the JSON file
#     """
#     # Find the JSON file in the directory
#     json_files = glob.glob(os.path.join(input_directory, '*.json'))
    
#     if not json_files:
#         print(f"No JSON files found in directory: {input_directory}")
#         return
    
#     if len(json_files) > 1:
#         print(f"Multiple JSON files found in directory. Using the first one: {json_files[0]}")
    
#     json_file_path = json_files[0]
#     output_csv_path = os.path.splitext(json_file_path)[0] + '.csv'
    
#     # Read the JSON data from the file
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as json_file:
#             data = json.load(json_file)
#     except Exception as e:
#         print(f"Error reading JSON file: {e}")
#         return

#     # Initialize CSV writer
#     output = StringIO()
#     csv_writer = csv.writer(output)
#     headers_written = False

#     # Process each page
#     for page_num, page_data in data['results'].items():
#         try:
#             page_content = json.loads(page_data)
            
#             for item in page_content:
#                 if item['category'] == 'Table':
#                     soup = BeautifulSoup(item['text'], 'html.parser')
#                     table = soup.find('table')
                    
#                     # Get headers if not already written
#                     if not headers_written and table:
#                         headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]
#                         csv_writer.writerow(headers)
#                         headers_written = True
                    
#                     # Write rows
#                     if table:
#                         for row in table.find('tbody').find_all('tr'):
#                             row_data = [td.get_text(strip=True) for td in row.find_all('td')]
#                             csv_writer.writerow(row_data)
#         except Exception as e:
#             print(f"Error processing page {page_num}: {e}")
#             continue

#     # Get the CSV content and save to file
#     csv_content = output.getvalue()
#     try:
#         with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#             csv_file.write(csv_content)
#         print(f"Successfully saved extracted table data to {output_csv_path}")
#     except Exception as e:
#         print(f"Error saving output CSV file: {e}")

# # Example usage:
# # input_directory = 'output'  # Replace with your directory path
# # process_bank_statement_json_to_csv(input_directory)

import json
from bs4 import BeautifulSoup
import csv
from io import StringIO
import os
import glob
import re

def process_bank_statement_json_to_csv(input_directory):
    json_files = glob.glob(os.path.join(input_directory, '*.json'))
    if not json_files:
        print(f"No JSON files found in directory: {input_directory}")
        return
    
    json_file_path = json_files[0]
    output_csv_path = os.path.splitext(json_file_path)[0] + '.csv'
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    output = StringIO()
    csv_writer = csv.writer(output)
    headers_written = False

    sorted_pages = sorted(
        data['results'].items(),
        key=lambda x: int(re.search(r'(\d+)', x[0]).group())
    )

    for page_num, page_data in sorted_pages:
        try:
            # Ensure JSON string is valid
            page_content = json.loads(page_data)
        except Exception as e:
            print(f"⚠️ Skipping {page_num}: Invalid JSON ({e})")
            continue

        for item in page_content:
            if item['category'] == 'Table':
                try:
                    soup = BeautifulSoup(item['text'], 'html.parser')
                    table = soup.find('table')

                    if not table:  # No table found
                        print(f"⚠️ Skipping {page_num}: No table detected")
                        continue

                    # Write headers once
                    if not headers_written:
                        thead = table.find('thead')
                        if thead:
                            headers = [th.get_text(strip=True) for th in thead.find_all('th')]
                            csv_writer.writerow(headers)
                            headers_written = True

                    # Write table rows
                    tbody = table.find('tbody')
                    if tbody:
                        for row in tbody.find_all('tr'):
                            row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                            csv_writer.writerow(row_data)
                except Exception as e:
                    print(f"⚠️ Skipping {page_num}: Error processing table ({e})")
                    continue

    csv_content = output.getvalue()
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_file.write(csv_content)
        print(f"✅ Successfully saved extracted table data to {output_csv_path}")
    except Exception as e:
        print(f"Error saving output CSV file: {e}")
# input_directory = 'output'  # Replace with your directory path
# process_bank_statement_json_to_csv(input_directory)