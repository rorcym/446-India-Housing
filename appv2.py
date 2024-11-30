import streamlit as st
import pandas as pd
import joblib
import base64

# Load the trained Random Forest model and scaler
model = joblib.load("rf_fixed.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset
file_path = "India LatiLongi.csv"
data = pd.read_csv(file_path)

# Page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide", initial_sidebar_state="auto")

# Add company logo in the corner
logo_path = "compressed.png"
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()

# CSS for styling, including forcing metric text to be black
st.markdown(
    f"""
    <style>
        /* Overall App Styling */
        [data-testid="stAppViewContainer"] {{
            background-color: #f9f9f9;  /* Light background */
        }}
        [data-testid="stSidebar"] {{
            background-color: #ffffff;  /* White sidebar */
        }}
        h1, h2, h3 {{
            color: #000000;  /* Black headings */
            font-weight: bold;
        }}
        label {{
            color: #000000 !important;  /* Black labels */
            font-size: 14px;
            font-weight: bold;
        }}
        input, select {{
            background-color: #ffffff !important;  /* Light selectors */
            color: #000000 !important;
            border: 1px solid #cccccc !important;
            border-radius: 5px;
            padding: 8px;
        }}
        .stSlider > div:first-child {{
            color: #000000 !important;  /* Black slider label */
        }}
        .stSlider > div:nth-child(2) .css-qrbaxs {{
            background-color: #1975bb !important; /* Blue slider fill */
        }}
        .stSlider > div:nth-child(2) .css-1rxz5jq {{
            color: #000000 !important;  /* Black slider text */
        }}
        /* Tabs Styling */
        .stTabs [role="tab"] {{
            background-color: #f2f2f2; /* Light gray for unselected tabs */
            color: #6e6e6e; /* Gray text */
            font-size: 16px;
            padding: 10px;
        }}
        .stTabs [role="tab"]:hover {{
            background-color: #e6e6e6; /* Slightly darker gray */
        }}
        .stTabs [role="tab"][aria-selected="true"] {{
            background-color: #ffffff; /* White for selected tab */
            color: #1975bb; /* Blue text for selected tab */
            border-bottom: 3px solid #1975bb;
            font-weight: bold;
        }}
        .stButton>button {{
            background-color: #1975bb; /* Primary blue */
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stButton>button:hover {{
            background-color: #145d91;  /* Darker blue */
        }}
        /* Improved Predicted Price Display */
        .result-container {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #000000; /* Black text */
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;  /* White background */
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }}
        /* Force black text for st.metric numbers */
        [data-testid="stMetricValue"] {{
            color: #000000 !important;  /* Set to black */
        }}
        [data-testid="stMetricLabel"] {{
            color: #000000 !important;  /* Ensure labels are also black */
        }}
    </style>
    <div class="logo-container" style="position: absolute; top: 10px; right: 20px;">
        <img src="data:image/png;base64,{encoded_logo}" width="200"/>
    </div>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("🏡 Indian House Price Prediction App")

# Tabs for functionality
tabs = st.tabs(["Enter the details of the house to predict its price", 
                "Compare Actual vs Fair Price for Dataset Houses"])

# Tab 1: Custom House Prediction
with tabs[0]:
    st.markdown("### Enter the details of the house to predict its price:")

    # Input fields for house features, divided into two columns
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=1.0, max_value=5.0, value=2.0)
        living_area = st.number_input("🏠 Living Area (sq ft)", min_value=500, max_value=10000, value=2000)
        waterfront = st.selectbox("🌊 Waterfront Present?", ["No", "Yes"], index=0)

    with col2:
        distance = st.number_input("✈️ Distance from the Airport (km)", min_value=1, max_value=100, value=10)
        schools = st.number_input("🏫 Number of Schools Nearby", min_value=0, max_value=10, value=2)
        condition = st.slider("🏚️ Condition of the House (1-5)", min_value=1, max_value=5, value=3)
        latitude = st.number_input("🌍 Latitude", min_value=-90.0, max_value=90.0, value=52.7609)
        longitude = st.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=-114.418)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'number of bedrooms': [bedrooms],
        'number of bathrooms': [bathrooms],
        'living area': [living_area],
        'waterfront present': [1 if waterfront == "Yes" else 0],
        'condition of the house': [condition],
        'Distance from the airport': [distance],
        'Number of schools nearby': [schools],
        'Lattitude': [latitude],
        'Longitude': [longitude]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict Price Button
    if st.button("Predict Price"):
        predicted_price = model.predict(input_data_scaled)[0]
        st.markdown(f"""
            <div class="result-container">
                💰 Predicted House Price: ₹{predicted_price:,.2f}
            </div>
        """, unsafe_allow_html=True)

# Tab 2: Inspect Dataset Houses
with tabs[1]:
    st.markdown("### Compare Actual vs Fair Price for Dataset Houses")

    # House ID selection
    house_id = st.number_input("Enter the House ID:", value=int(data.iloc[0]['id']), min_value=int(data['id'].min()), max_value=int(data['id'].max()))

    # Filter the dataset by house ID
    selected_house = data[data['id'] == house_id]
    if selected_house.empty:
        st.error("❌ House ID not found. Please try another.")
    else:
        st.markdown("### Selected House Details")
        st.dataframe(selected_house)

        # Prepare data for prediction
        input_features = selected_house[['number of bedrooms', 'number of bathrooms', 'living area',
                                         'waterfront present', 'condition of the house',
                                         'Distance from the airport', 'Number of schools nearby',
                                         'Lattitude', 'Longitude']]
        input_features_scaled = scaler.transform(input_features)

        # Model prediction
        predicted_price = model.predict(input_features_scaled)[0]
        actual_price = selected_house['Price'].values[0]

        # Display comparison with improved styling
        col1, col2 = st.columns(2)
        col1.metric("Actual Price", f"₹{actual_price:,.2f}")
        col2.metric("Fair Price (Model)", f"₹{predicted_price:,.2f}")

        # Overvaluation or undervaluation message
        difference = actual_price - predicted_price
        if difference > 0:
            st.markdown(f"""
                <div class="result-container" style="background-color: #ffe6e6;">
                    💸 The house is overpriced by: ₹{difference:,.2f}
                </div>
            """, unsafe_allow_html=True)
        elif difference < 0:
            st.markdown(f"""
                <div class="result-container" style="background-color: #e6ffe6;">
                    🎉 The house is undervalued by: ₹{abs(difference):,.2f}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("✨ The house is priced fairly.")
