import pandas as pd
import os
import time
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Ensure Streamlit maintains execution context
st.experimental_set_query_params(refresh="true")

# Get current date and timestamp
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, key="fizzbuzzcounter")

# Display FizzBuzz Logic
if count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Attendance CSV File Path
csv_file_path = f"Attendance/Attendance_{date}.csv"

# Display header with current date
st.header(f"Attendance Record for {date}")

if os.path.exists(csv_file_path):
    # Check CSV header format
    with open(csv_file_path, 'r') as f:
        header_line = f.readline().strip()
        header_fields = header_line.split(',')
    
    expected_columns = ['NAME', 'DATE', 'TIME']
    
    # If file is in old format with 2 columns, update it automatically
    if len(header_fields) == 2 and header_fields == ['NAME', 'TIME']:
        try:
            df_old = pd.read_csv(csv_file_path)
            # Add DATE column with the current date for each row
            df_old['DATE'] = date
            # Reorder columns to match the expected format
            df_new = df_old[['NAME', 'DATE', 'TIME']]
            df_new.to_csv(csv_file_path, index=False)
            st.info("CSV file updated to new format (added DATE column).")
        except Exception as e:
            st.error(f"Error updating CSV file: {e}")
    elif len(header_fields) != len(expected_columns):
        st.error(
            f"CSV file format error: Expected {len(expected_columns)} columns {expected_columns} "
            f"but found {len(header_fields)} columns: {header_fields}. "
            "Please update or delete the file to use the new format."
        )

    # Try reading the CSV file with retries
    retry = 3  # Number of retry attempts
    for attempt in range(retry):
        try:
            df = pd.read_csv(csv_file_path)
            # Search Filter Input
            search_query = st.text_input("Search Attendance Record:", "")
            # Filter DataFrame based on search query
            if search_query:
                filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)]
            else:
                filtered_df = df
            # Display DataFrame with highlighted max values
            st.dataframe(filtered_df.style.highlight_max(axis=0))
            break  # Successfully read and displayed the file
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)  # Wait and retry
            else:
                st.error(f"Error reading the CSV file: {e}")
else:
    st.warning(f"No attendance record found for {date}.")
