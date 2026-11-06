
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

model = load_model()

# --------------------------------
# SESSION STATE
# --------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

def navigate(page):
    st.session_state.page = page


# --------------------------------
# DASHBOARD
# --------------------------------

if st.session_state.page == "Dashboard":

    st.title("📩 AI Spam Message Detector")

    st.write(
        "An intelligent SMS classification system "
        "using Machine Learning and NLP."
    )

    st.divider()

    st.header("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "98.30%")
    col2.metric("Precision", "97.10%")
    col3.metric("Recall", "89.90%")
    col4.metric("F1-Score", "93.40%")

    st.success("Best Performing Model: Support Vector Machine (SVM)")

    st.divider()

    st.header("🚀 Choose Detection Mode")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🔍 Single SMS Detection")

            st.write(
                "Enter one SMS message and check whether "
                "it is Spam or Normal."
            )

            if st.button(
                "Open Single Detection →",
                use_container_width=True
            ):
                navigate("Single Detection")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("📂 Bulk SMS Detection")

            st.write(
                "Upload a CSV file to classify multiple "
                "SMS messages at once."
            )

            if st.button(
                "Open Bulk Detection →",
                use_container_width=True
            ):
                navigate("Bulk Detection")
                st.rerun()

# --------------------------------
# SINGLE DETECTION PAGE
# --------------------------------

elif st.session_state.page == "Single Detection":

    st.title("🔍 Single SMS Detection")

    if st.button("← Back to Dashboard"):
        navigate("Dashboard")
        st.rerun()

    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Enter your message here..."
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
# BULK DETECTION PAGE
# --------------------------------

elif st.session_state.page == "Bulk Detection":

    st.title("📂 Bulk SMS Detection")

    if st.button("← Back to Dashboard"):
        navigate("Dashboard")
        st.rerun()

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
                st.error("CSV must contain a 'message' column.")

            elif data.empty:
                st.warning("The CSV file is empty.")

            else:
                messages = data["message"].fillna("").astype(str)

                if (messages.str.strip() == "").all():
                    st.warning("No valid messages found.")

                else:
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

                    st.subheader("Classification Results")

                    st.dataframe(
                        data,
                        use_container_width=True
                    )

                    chart_data = pd.DataFrame({
                        "Category": ["Spam", "Normal"],
                        "Count": [spam_count, normal_count]
                    })

                    st.subheader("Spam vs Normal Messages")

                    st.bar_chart(
                        chart_data.set_index("Category")
                    )

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
            st.error(f"Error processing CSV: {e}")
