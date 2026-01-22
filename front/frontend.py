import os
from typing import List

import pandas as pd
import requests
import streamlit as st


def get_backend_url():
    """Get the URL of the backend service."""
    # For local development, use environment variable or default
    return os.environ.get("BACKEND_URL", "http://localhost:8000")


def predict_fraud(features: List[float], backend_url: str):
    """Send the transaction features to the backend for fraud prediction."""
    predict_url = f"{backend_url}/predict"

    # Prepare the request data
    data = {"features": features}

    try:
        response = requests.post(predict_url, json=data, headers={"Content-Type": "application/json"}, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def create_sample_transaction():
    """Create a sample transaction for testing."""
    # Generate realistic credit card transaction features
    # Based on typical V1-V28 PCA features from credit card dataset
    sample_features = [
        -1.3598071336738,
        -0.0727811733098497,
        2.53634673796914,
        1.37815522427443,
        -0.338320769942518,
        0.462387777762292,
        0.239598554061257,
        0.0986979012610507,
        0.363786969611213,
        0.0907941719789316,
        -0.551599533260813,
        -0.617800855762348,
        -0.991389847235408,
        -0.311169353699879,
        1.46817697209427,
        -0.470400525259478,
        0.207971241929242,
        0.0257905801985591,
        0.403992960255733,
        0.251412098239705,
        -0.018306777944153,
        0.277837575558899,
        -0.110473910188767,
        0.0669280749146731,
        0.128539358273528,
        -0.189114843888824,
        0.133558376740387,
        -0.0210530534538215,
    ]
    return sample_features


def main():
    """Main function of the Streamlit frontend."""
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")

    # Get backend URL
    backend_url = get_backend_url()

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            """
        This system uses an autoencoder neural network to detect fraudulent credit card transactions.

        **How it works:**
        - The model reconstructs transaction features
        - High reconstruction error indicates potential fraud
        - Uses anomaly detection principles
        """
        )

        st.header("🔗 Backend Status")
        try:
            response = requests.get(f"{backend_url}/docs", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend Connected")
            else:
                st.error("❌ Backend Issue")
        except Exception:
            st.error("❌ Backend Disconnected")

        st.write(f"**Backend URL:** `{backend_url}`")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🔍 Transaction Analysis")

        # Input method selection
        input_method = st.radio("Choose input method:", ["Manual Entry", "Sample Transaction", "Upload CSV"])

        features = []

        if input_method == "Manual Entry":
            st.subheader("Enter Transaction Features (V1-V28)")

            # Create input fields for 28 features
            cols = st.columns(4)
            features = []

            for i in range(28):
                with cols[i % 4]:
                    feature_value = st.number_input(
                        f"V{i + 1}",
                        value=0.0,
                        format="%.6f",
                        key=f"feature_{i}",
                        help=f"Feature V{i + 1} - PCA transformed feature",
                    )
                    features.append(feature_value)

        elif input_method == "Sample Transaction":
            st.subheader("Sample Transaction Data")
            features = create_sample_transaction()

            # Display sample features in a nice format
            feature_df = pd.DataFrame({"Feature": [f"V{i + 1}" for i in range(28)], "Value": features})

            # Split into multiple columns for better display
            cols = st.columns(3)
            for i, col in enumerate(cols):
                start_idx = i * 10
                end_idx = min((i + 1) * 10, 28)
                with col:
                    st.dataframe(feature_df.iloc[start_idx:end_idx], hide_index=True, use_container_width=True)

        elif input_method == "Upload CSV":
            st.subheader("Upload Transaction CSV")
            uploaded_file = st.file_uploader(
                "Choose a CSV file with transaction features",
                type="csv",
                help="CSV should have 28 columns (V1-V28) representing transaction features",
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)

                    if len(df.columns) != 28:
                        st.error(f"CSV must have exactly 28 columns, got {len(df.columns)}")
                    else:
                        st.success(f"Loaded {len(df)} transactions")

                        # Select which transaction to analyze
                        if len(df) > 1:
                            selected_row = st.selectbox(
                                "Select transaction to analyze:",
                                range(len(df)),
                                format_func=lambda x: f"Transaction {x + 1}",
                            )
                            features = df.iloc[selected_row].tolist()
                        else:
                            features = df.iloc[0].tolist()

                        # Display selected transaction
                        st.write("Selected transaction features:")
                        feature_df = pd.DataFrame({"Feature": [f"V{i + 1}" for i in range(28)], "Value": features})
                        st.dataframe(feature_df.T, use_container_width=True)

                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")

        # Prediction button
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            if len(features) == 28:
                with st.spinner("Analyzing transaction..."):
                    result = predict_fraud(features, backend_url)

                if result:
                    # Store result in session state for the results column
                    st.session_state["prediction_result"] = result
            else:
                st.error("Please provide all 28 transaction features")

    with col2:
        st.header("📊 Results")

        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]

            # Main prediction result
            is_fraud = result.get("is_fraud", False)
            reconstruction_error = result.get("reconstruction_error", 0.0)
            threshold = result.get("threshold", 0.005)

            # Display result with appropriate styling
            if is_fraud:
                st.error("🚨 **FRAUD DETECTED**")
                st.markdown(
                    (
                        '<div style="background-color: #ffebee; padding: 15px; '
                        'border-radius: 5px; border-left: 4px solid #f44336;">'
                        '<h4 style="color: #d32f2f; margin: 0;">'
                        "High Risk Transaction"
                        "</h4>"
                        '<p style="margin: 5px 0 0 0;">'
                        "This transaction shows suspicious patterns"
                        "</p>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
                st.markdown(
                    (
                        '<div style="background-color: #e8f5e8; padding: 15px; '
                        'border-radius: 5px; border-left: 4px solid #4caf50;">'
                        '<h4 style="color: #388e3c; margin: 0;">'
                        "Normal Transaction</h4>"
                        '<p style="margin: 5px 0 0 0;">'
                        "This transaction appears legitimate"
                        "</p>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            # Detailed metrics
            st.subheader("📈 Detailed Analysis")

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Reconstruction Error",
                    f"{reconstruction_error:.6f}",
                    delta=f"{reconstruction_error - threshold:.6f}" if reconstruction_error > threshold else None,
                    delta_color="inverse",
                )

            with col_b:
                st.metric("Fraud Threshold", f"{threshold:.6f}")

            # Visualization
            st.subheader("📊 Error Analysis")

            # Create a gauge-like visualization
            chart_data = pd.DataFrame(
                {
                    "Metric": ["Reconstruction Error", "Threshold"],
                    "Value": [reconstruction_error, threshold],
                    "Type": ["Actual", "Threshold"],
                }
            )

            st.bar_chart(chart_data.set_index("Metric")["Value"])

            # Risk level indicator
            if reconstruction_error > threshold * 2:
                risk_level = "🔴 VERY HIGH RISK"
                risk_color = "#d32f2f"
            elif reconstruction_error > threshold:
                risk_level = "🟡 MODERATE RISK"
                risk_color = "#f57c00"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "#388e3c"

            st.markdown(
                (
                    f'<div style="text-align: center; padding: 10px; '
                    f"background-color: {risk_color}20; "
                    'border-radius: 5px; margin-top: 10px;">'
                    f'<h3 style="color: {risk_color}; margin: 0;">'
                    f"{risk_level}"
                    "</h3>"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("👆 Analyze a transaction to see results here")

            # Show some helpful information
            st.markdown(
                """
            **What you'll see here:**
            - 🎯 Fraud prediction (Fraud/Legitimate)
            - 📊 Reconstruction error value
            - 🎚️ Threshold comparison
            - 📈 Risk level assessment
            """
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 14px;'>"
        "Credit Card Fraud Detection System | Powered by PyTorch Lightning & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
