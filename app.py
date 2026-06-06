import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Set the basic page configuration
st.set_page_config(page_title="Shopper Spectrum", layout="wide", page_icon="🛒")

# 2. Load the machine learning models safely
@st.cache_resource
def load_models():
    # We use a relative path since the models folder is in the same directory as app.py
    model_path = "models"
    
    kmeans = joblib.load(os.path.join(model_path, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    similarity = joblib.load(os.path.join(model_path, 'similarity_matrix.pkl'))
    
    return kmeans, scaler, similarity

# Call the function to actually load them into memory
try:
    kmeans_final, scaler, similarity_matrix = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# 3. Build the Sidebar Navigation
st.sidebar.title("Shopper Spectrum")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation Menu", ["Home", "Product Recommender", "Customer Segmentation"])

# 4. Build the Home Page (Placeholder)
if app_mode == "Home":
    st.title("🛒 Welcome to Shopper Spectrum")
    if models_loaded:
        st.success("All Machine Learning models successfully loaded and ready!")
    st.write("Please use the sidebar menu to navigate between the analytical modules.")

elif app_mode == "Product Recommender":
    st.title("Product Recommender")
    
    # 1. Text input box for Product Name (defaulting to the client's test case)
    product_input = st.text_input("Enter Product Name", "GREEN VINTAGE SPOT BEAKER")
    
    # --- The "Cheat Sheet" Expander ---
    with st.expander("Need ideas? Click here for popular products"):
        st.markdown("""
        Try copying and pasting one of our all-time best sellers:
        * WHITE HANGING HEART T-LIGHT HOLDER
        * JUMBO BAG RED RETROSPOT
        * REGENCY CAKESTAND 3 TIER
        * PARTY BUNTING
        * LUNCH BAG RED RETROSPOT
        """)

    # 2. Button: Get Recommendations
    if st.button("Recommend"):
        # First, we must check if the product actually exists in our database
        if product_input in similarity_matrix.columns:
            # Sort similarities, skip index 0 (itself), and grab the top 5
            recommendations = similarity_matrix[product_input].sort_values(ascending=False).iloc[1:6]
            
            # 3. Display 5 recommended products as a styled list
            st.markdown("**Recommended Products:**")
            for item in recommendations.index:
                # Using st.info creates a nice "card" view as requested in the document
                st.info(item) 
        else:
            st.error("Product not found in our database. Please check the exact spelling.")

elif app_mode == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write("Enter a customer's shopping history to predict their segment.")

    # 1. Create the input fields (defaulting to the test case from the project doc)
    recency = st.number_input("Recency (days since last purchase)", min_value=1, value=325)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=765.00, step=1.0)

    # 2. Button: Predict Cluster
    if st.button("Predict Segment"):
        
        # Step A: Put the raw inputs into a dataframe (matching original column names)
        input_data = pd.DataFrame([[recency, frequency, monetary]], 
                                  columns=['Recency', 'Frequency', 'Monetary'])
        
        # Step B: The Invisible Pipeline (Log transform, then Scale!)
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        
        # Step C: Ask the K-Means model to predict the cluster number (0, 1, 2, or 3)
        cluster_prediction = kmeans_final.predict(input_scaled)[0]
        
        # Step D: Map the math back to our business labels
        cluster_mapping = {
            0: 'High-Value',
            1: 'At-Risk',
            2: 'Occasional',
            3: 'Regular'
        }
        segment_label = cluster_mapping[cluster_prediction]
        
        # Step E: Display the result cleanly to the user
        st.markdown("---")
        st.metric(label="Predicted Cluster", value=str(cluster_prediction))
        st.success(f"This customer belongs to: **{segment_label}**")