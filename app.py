import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model = pkl.load(open("smart_phone_model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Trained model file 'smart_phone_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Title of the app
st.title("📱 Smartphone Price Prediction")
st.write("Fill in the smartphone specifications to predict the price.")

# Sidebar for user input
st.sidebar.header("Input Smartphone Specifications")

def user_input_features():
    brand = st.sidebar.selectbox("Brand", ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Vivo", "Oppo", "Motorola", "Nokia"])
    processor = st.sidebar.selectbox("Processor", ["snapdragon", "exynos", "mediatek", "kirin", "apple"])
    os = st.sidebar.selectbox("Operating System", ["android", "ios"])

    RAM = st.sidebar.slider("RAM (GB)", 2, 16, 8)
    ROM = st.sidebar.slider("ROM (GB)", 32, 512, 128)
    battery = st.sidebar.slider("Battery Capacity (mAh)", 2000, 6000, 4500)
    screen_size_inches = st.sidebar.slider("Screen Size (inches)", 4.5, 7.0, 6.5)
    rear_camera_pixel = st.sidebar.slider("Rear Camera (MP)", 8, 108, 48)
    front_camera_pixel = st.sidebar.slider("Front Camera (MP)", 5, 32, 16)
    num_rear_cameras = st.sidebar.slider("Number of Rear Cameras", 1, 4, 3)
    num_front_cameras = st.sidebar.slider("Number of Front Cameras", 1, 2, 1)
    has_5g = st.sidebar.radio("5G Support", ["Yes", "No"])

    data = {
        "brand": brand,
        "processor": processor,
        "os": os,
        "RAM": RAM,
        "ROM": ROM,
        "battery": battery,
        "screen_size_inches": screen_size_inches,
        "rear_camera_pixel": rear_camera_pixel,
        "front_camera_pixel": front_camera_pixel,
        "num_rear_cameras": num_rear_cameras,
        "num_front_cameras": num_front_cameras,
        "5G": 1 if has_5g == "Yes" else 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Scale numerical features
scaler = StandardScaler()
numerical_features = ["RAM", "ROM", "battery", "screen_size_inches", "rear_camera_pixel", "front_camera_pixel", "num_rear_cameras", "num_front_cameras"]
input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])

# Encoding categorical features
brand_list = ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Vivo", "Oppo", "Motorola", "Nokia"]
processor_list = ["snapdragon", "exynos", "mediatek", "kirin", "apple"]
os_list = ["android", "ios"]

def encode_feature(value, feature_list, prefix):
    return {f"{prefix}_{item}": 1 if item == value else 0 for item in feature_list}

brand_encoded = encode_feature(input_df["brand"].iloc[0], brand_list, "brand")
processor_encoded = encode_feature(input_df["processor"].iloc[0], processor_list, "processor")
os_encoded = encode_feature(input_df["os"].iloc[0], os_list, "os")

# Combine all features
final_input = pd.concat([
    input_df.drop(columns=["brand", "processor", "os"]),
    pd.DataFrame([brand_encoded]),
    pd.DataFrame([processor_encoded]),
    pd.DataFrame([os_encoded])
], axis=1)

# Ensure final input matches model features
if hasattr(model, 'feature_names_in_'):
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in final_input.columns:
            final_input[feature] = 0
    final_input = final_input[model_features]
else:
    st.error("❌ Model does not contain 'feature_names_in_'. Please retrain with sklearn 1.0 or above.")
    st.stop()

# Prediction button
if st.button("🔮 Predict Price"):
    try:
        prediction = model.predict(final_input)
        st.success(f"💵 Predicted Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

st.write("---")
st.write("Developed using Streamlit and RandomForestRegressor. 🚀")
