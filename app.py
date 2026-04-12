import streamlit as st
import pandas as pd
import os
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_transformation import DatetimeIndexer, TimeSeriesImputer, TimeSeriesResampler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Power Consumption Forecaster",
    page_icon="⚡",
    layout="wide"
)

# --- HEADER SECTION ---
st.title("⚡ Power Consumption Forecasting App")
st.markdown("""
Welcome to the Energy Forecasting Dashboard! 
Upload your most recent historical energy data, and our automated Machine Learning pipeline 
will forecast the upcoming power consumption trends.
""")
st.divider()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Upload Recent Historical Data (CSV)", type=["csv", "txt"])

forecast_steps = st.sidebar.slider(
    "2. Select Forecast Horizon (Hours)", 
    min_value=12, 
    max_value=168, # Up to one week (24 * 7)
    value=24, 
    step=12
)

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # 1. Preview the uploaded data
    raw_df = pd.read_csv(uploaded_file, sep=';', na_values=['?'], low_memory=False) # Adjust separator if needed based on your raw data
    
    with st.expander("Preview Uploaded Data"):
        st.dataframe(raw_df.tail(5)) # Show the most recent rows

    # 2. Prediction Trigger
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner('Initializing ML Pipeline & Generating Predictions...'):
            try:
                # Save uploaded file temporarily for the pipeline to read
                temp_file_path = os.path.join('artifacts', 'temp_upload.csv')
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                
                # Save exactly as it was uploaded to maintain format for the preprocessor
                raw_df.to_csv(temp_file_path, index=False, header=True, sep=';', na_rep='?')

                # Initialize and run the prediction pipeline
                pipeline = PredictPipeline()
                forecast_df = pipeline.predict(recent_data_path=temp_file_path,
                                               steps_to_forecast=forecast_steps)

                st.divider()
                st.subheader(f"🔮 Forecast for the next {forecast_steps} hours")

                # --- VISUALIZATION SECTION ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Beautiful native line chart
                    st.line_chart(forecast_df['Forecast'], color="#FF4B4B")

                with col2:
                    # Show the actual numbers
                    st.dataframe(forecast_df, use_container_width=True)
                
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
else:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
    
    # Provide a tiny dummy dataset layout so the user knows what to upload
    st.markdown("### Expected Data Format:")
    st.markdown("Your uploaded CSV should match the format of your training data (containing `Date`, `Time`, and `Global_active_power`).")