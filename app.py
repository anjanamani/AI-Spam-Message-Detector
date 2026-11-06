
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="📩",
    layout="wide"
)

# Load model
model = joblib.load("spam_model.pkl")

# Application heading
st.title("📩 AI Spam Message Detector")
st.write("An intelligent SMS classification system using Machine Learning.")

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a feature",
    ["Dashboard", "Single Detection", "Bulk Detection"]
)

# --------------------------------
# DASHBOARD
# --------------------------------

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    st.write(
        "Upload messages or enter an SMS to identify potential spam."
    )

    st.info(
        "This application uses a trained machine learning model "
        "to classify messages as Spam or Normal."
    )

    st.subheader("Machine Learning Algorithms")

    st.write("- Multinomial Naive Bayes")
    st.write("- Logistic Regression")
    st.write("- Support Vector Machine (SVM)")

    st.subheader("Project Features")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Modes", "2")
    col2.metric("ML Algorithms Compared", "3")
    col3.metric("Output Classes", "2")

# --------------------------------
# SINGLE DETECTION
# --------------------------------

elif page == "Single Detection":

    st.header("🔍 Single Message Detection")

    message = st.text_area("Enter your SMS message:")

    if st.button("Check Message"):

        if message.strip() == "":
            st.warning("Please enter a message.")

        else:
            prediction = model.predict([message])[0]

            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            spam_index = classes.index(1)

            spam_probability = probabilities[spam_index] * 100

            if prediction == 1:
                st.error("⚠️ This is a SPAM message!")
            else:
                st.success("✅ This is a NORMAL message!")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

# --------------------------------
# BULK DETECTION
# --------------------------------

elif page == "Bulk Detection":

    st.header("📂 Bulk Message Detection")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        if "message" not in data.columns:
            st.error("CSV must contain a 'message' column.")

        elif data.empty:
            st.warning("The CSV file is empty.")

        else:
            predictions = model.predict(
                data["message"].fillna("").astype(str)
            )

            data["prediction"] = [
                "Spam" if value == 1 else "Normal"
                for value in predictions
            ]

            spam_count = (data["prediction"] == "Spam").sum()
            normal_count = (data["prediction"] == "Normal").sum()

            total = len(data)

            # Metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Messages", total)
            col2.metric("Spam Messages", spam_count)
            col3.metric("Normal Messages", normal_count)

            st.subheader("Classification Results")

            st.dataframe(data, use_container_width=True)

            # Chart
            chart_data = pd.DataFrame({
                "Category": ["Spam", "Normal"],
                "Count": [spam_count, normal_count]
            })

            st.subheader("Spam vs Normal Messages")

            st.bar_chart(
                chart_data.set_index("Category")
            )

            # Download
            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "spam_detection_results.csv",
                "text/csv"
            )
