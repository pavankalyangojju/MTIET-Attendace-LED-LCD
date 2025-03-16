import pandas as pd
import os
import time
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 2 seconds
st_autorefresh(interval=2000, key="refresh")

# Get current date
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Attendance CSV File Path
csv_file_path = f"Attendance/Attendance_{date}.csv"

# Display header with current date
st.header(f"Attendance Record for {date}")

if os.path.exists(csv_file_path):
    # Define expected columns
    expected_columns = ['NAME', 'RFID', 'DATE', 'TIME']

    # Check CSV header format
    with open(csv_file_path, 'r') as f:
        header_line = f.readline().strip()
        header_fields = header_line.split(',')

    # Ensure CSV has the correct format
    if len(header_fields) != len(expected_columns):
        st.error(
            f"CSV file format error: Expected {len(expected_columns)} columns {expected_columns} "
            f"but found {len(header_fields)} columns: {header_fields}. "
            "Please update or delete the file to use the correct format."
        )
    else:
        # Read CSV file
        df = pd.read_csv(csv_file_path)

        # Search filter
        search_query = st.text_input("Search Attendance Record:", "")
        if search_query:
            filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)]
        else:
            filtered_df = df

        # Display Data Table
        st.dataframe(filtered_df.style.highlight_max(axis=0))

        # Generate Attendance Count Per Person
        attendance_count = df["NAME"].value_counts()

        # Plot Graph
        st.subheader("Attendance Frequency Graph")
        fig, ax = plt.subplots()
        attendance_count.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_xlabel("Name")  # X-axis label as "Name"
        ax.set_ylabel(f"Attendance on {date}")  # Y-axis label with the date
        ax.set_title("Attendance Count Per Person")
        st.pyplot(fig)

else:
    st.warning(f"No attendance record found for {date}.")`
