import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 5 seconds to get the latest data
count = st_autorefresh(interval=5000, key="api_refresh")

# Set the Flask API endpoint
API_URL = "http://localhost:5000/attendance"

# Page title
st.title("?? Real-Time Attendance Dashboard")

# Get the current date
ts = time.time()
current_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Fetch attendance data from API
try:
    response = requests.get(API_URL)
    
    if response.status_code == 200:
        data = response.json()

        if data:
            df = pd.DataFrame(data)  # Convert JSON response to DataFrame

            # Search Filter Input
            search_query = st.text_input("?? Search Attendance:", "")

            # Filter DataFrame based on search query
            if search_query:
                filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)]
            else:
                filtered_df = df

            # Display the attendance records in a table
            st.dataframe(filtered_df.style.highlight_max(axis=0))

        else:
            st.warning("No attendance records found.")

    else:
        st.error("?? Failed to fetch data from API. Please check if the Flask server is running.")

except Exception as e:
    st.error(f"?? Error connecting to API: {e}")

# Footer message
st.markdown("?? Data updates automatically every 5 seconds.")
