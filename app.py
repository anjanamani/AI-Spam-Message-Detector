
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="📩",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# --------------------------------
# APPLICATION HEADING
# --------------------------------

st.title("📩 AI Spam Message Detector")

st.write(
    "An intelligent SMS classification system "
    "using Natural Language Processing and Machine Learning."
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

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
        "Detect spam messages using machine learning. "
        "Enter an SMS or upload a CSV file for classification."
    )

    st.success("Best Performing Model: Support Vector Machine (SVM)")

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "98.30%")
    col2.metric("Precision", "97.10%")
    col3.metric("Recall", "89.90%")
    col4.metric("F1-Score", "93.40%")

    st.divider()

    st.subheader("Algorithms Compared")

    st.write("- Multinomial Naive Bayes")
    st.write("- Logistic Regression")
    st.write("- Support Vector Machine (SVM)")

    st.subheader("Project Features")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Modes", "2")
    col2.metric("Algorithms Compared", "3")
    col3.metric("Output Classes", "2")

    st.info(
        "The SVM classifier was selected based on its "
        "Spam F1-score of 93.4%."
    )

# --------------------------------
# SINGLE DETECTION
# --------------------------------

elif page == "Single Detection":

    st.header("🔍 Single Message Detection")

    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Example: Congratulations! You have won a free prize."
    )

    if st.button("Check Message", type="primary"):

        if not message.strip():

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

            st.progress(float(spam_probability / 100))

# --------------------------------
# BULK DETECTION
# --------------------------------

elif page == "Bulk Detection":

    st.header("📂 Bulk Message Detection")

    st.write(
        "Upload a CSV file containing a column named 'message'."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            if "message" not in data.columns:

                st.error(
                    "CSV must contain a column named 'message'."
                )

            elif data.empty:

                st.warning("The CSV file is empty.")

            else:

                # Handle missing messages
                messages = (
                    data["message"]
                    .fillna("")
                    .astype(str)
                )

                if (messages.str.strip() == "").all():

                    st.warning("No valid messages found.")

                else:

                    # Predictions
                    predictions = model.predict(messages)

                    probabilities = model.predict_proba(messages)

                    classes = list(model.classes_)
                    spam_index = classes.index(1)

                    spam_probabilities = (
                        probabilities[:, spam_index] * 100
                    )

                    data["prediction"] = [
                        "Spam" if value == 1 else "Normal"
                        for value in predictions
                    ]

                    data["spam_probability"] = [
                        round(value, 2)
                        for value in spam_probabilities
                    ]

                    # Metrics
                    spam_count = (
                        data["prediction"] == "Spam"
                    ).sum()

                    normal_count = (
                        data["prediction"] == "Normal"
                    ).sum()

                    total = len(data)

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Messages", total)
                    col2.metric("Spam Messages", spam_count)
                    col3.metric("Normal Messages", normal_count)

                    # Results
                    st.subheader("Classification Results")

                    st.dataframe(
                        data,
                        use_container_width=True
                    )

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
                    csv = data.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "spam_detection_results.csv",
                        "text/csv"
                    )

        except Exception as e:

            st.error(f"Error processing CSV file: {e}")
