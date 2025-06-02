import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="ML Workflow App", layout="wide")

# ---------------------- LOGIN SYSTEM ----------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():
    with st.form("login_form"):
        st.subheader("🔐 Login to Access the App")
        user_id = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if user_id == "admin" and password == "1234":
                st.success("✅ Login successful!")
                st.session_state["logged_in"] = True
            else:
                st.error("❌ Invalid credentials")

# ---------------------- SIDEBAR NAVIGATION ----------------------
st.sidebar.title("📊 Machine Learning App")
menu = st.sidebar.radio("Go to", ["Home", "Login", "Upload Data", "Preprocessing", "Visualization", "Linear Regression", "Logistic Regression", "Prediction"])

# ---------------------- HOME PAGE ----------------------
if menu == "Home":
    st.title("🏠 Welcome to ML Workflow App")
    st.markdown("""
    This application allows you to perform a complete machine learning workflow through a user-friendly interface.

    ### 🔧 Modules Included:
    - 📁 Upload CSV
    - 🧹 Data Preprocessing (Missing Values, Outliers, Encoding)
    - 📊 Data Visualization (Scatter, Bar, Count)
    - 📈 Linear Regression
    - 📈 Logistic Regression
    - 🔮 Prediction Interface (Dynamic Model Loading)
    """)

# ---------------------- LOGIN PAGE ----------------------
if menu == "Login":
    login()
    if st.session_state["logged_in"]:
        st.success("You are already logged in.")

# ---------------------- FILE UPLOAD ----------------------
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if menu == "Upload Data":
    if not st.session_state["logged_in"]:
        st.warning("🔒 Please login first.")
    else:
        st.title("📁 Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write("### 📄 Preview of Uploaded Data", df.head())
            st.session_state["raw_data"] = df
            st.session_state["preprocessed_data"] = df.copy()

# ---------------------- PREPROCESSING ----------------------
if menu == "Preprocessing":
    if not st.session_state["logged_in"]:
        st.warning("🔒 Please login first.")
    else:
        st.title("🧹 Data Preprocessing")
        if "raw_data" not in st.session_state:
            st.warning("⚠️ Please upload a dataset first from the 'Upload Data' tab.")
        else:
            df = st.session_state.get("preprocessed_data", st.session_state["raw_data"]).copy()
            tab1, tab2, tab3 = st.tabs(["🧹 Missing Value Treatment", "📏 Outlier Treatment", "🔡 Encoding"])

            with tab1:
                st.subheader("🔍 Missing Value Summary")
                missing_summary = df.isnull().sum()
                missing_summary = missing_summary[missing_summary > 0]
                if missing_summary.empty:
                    st.success("✅ No missing values detected!")
                else:
                    st.write(missing_summary)
                    cols_with_na = list(missing_summary.index)
                    selected_col = st.selectbox("Select a column to treat", cols_with_na)
                    if df[selected_col].dtype in ['float64', 'int64']:
                        strategy = st.radio("Choose strategy", ["Drop Rows", "Fill with Mean", "Fill with Median"])
                    else:
                        strategy = st.radio("Choose strategy", ["Drop Rows", "Fill with Mode"])
                    if st.button("Apply Missing Value Treatment"):
                        if strategy == "Drop Rows":
                            df = df[df[selected_col].notnull()]
                        elif strategy == "Fill with Mean":
                            df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
                        elif strategy == "Fill with Median":
                            df[selected_col] = df[selected_col].fillna(df[selected_col].median())
                        elif strategy == "Fill with Mode":
                            df[selected_col] = df[selected_col].fillna(df[selected_col].mode()[0])
                        st.session_state["preprocessed_data"] = df
                        st.success(f"✅ Treated missing values in `{selected_col}` using `{strategy}`")
                        st.write(df.head())

            with tab2:
                st.subheader("📏 Outlier Detection (IQR Method)")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if not numeric_cols:
                    st.info("No numeric columns available.")
                else:
                    selected_col = st.selectbox("Select numeric column", numeric_cols, key="outlier_col")
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    st.write(f"Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
                    st.write(f"Acceptable Range = [{lower:.2f}, {upper:.2f}]")
                    if st.button("Apply Outlier Filter"):
                        before = df.shape[0]
                        df = df[(df[selected_col] >= lower) & (df[selected_col] <= upper)]
                        after = df.shape[0]
                        st.session_state["preprocessed_data"] = df
                        st.success(f"✅ Removed {before - after} outlier rows.")
                        st.write(df.head())

            with tab3:
                st.subheader("🔡 Encoding Categorical Columns")
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if not cat_cols:
                    st.info("No categorical columns found.")
                else:
                    selected_col = st.selectbox("Select column to encode", cat_cols)
                    encoding = st.radio("Encoding Type", ["Label Encoding", "One-Hot Encoding"])
                    if st.button("Apply Encoding"):
                        if encoding == "Label Encoding":
                            df[selected_col] = df[selected_col].astype('category').cat.codes
                            st.success(f"✅ Label encoding applied to `{selected_col}`")
                        elif encoding == "One-Hot Encoding":
                            df = pd.get_dummies(df, columns=[selected_col], dtype=int, drop_first=True)
                            st.success(f"✅ One-hot encoding applied to `{selected_col}`")
                        st.session_state["preprocessed_data"] = df
                        st.write(df.head())

            st.subheader("⬇️ Export Cleaned Data")
            csv = st.session_state["preprocessed_data"].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="cleaned_data.csv", mime="text/csv")

# ---------------------- MODEL TRAINING ----------------------
if menu == "Linear Regression":
    if not st.session_state["logged_in"]:
        st.warning("🔒 Please login first.")
    else:
        st.title("📈 Linear Regression")
        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please preprocess and upload data first.")
        else:
            df = st.session_state["preprocessed_data"].copy()
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            target_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
            feature_cols = st.multiselect("Select Feature Columns (X)", [col for col in numeric_cols if col != target_col])

            if st.button("Train Model"):
                X = df[feature_cols]
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("Model Performance:")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
                st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
                st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                joblib.dump(model, "linear_model.pkl")
                joblib.dump(feature_cols, "features.pkl")
                with open("linear_model.pkl", "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="linear_model.pkl")
                with open("features.pkl", "rb") as f:
                    st.download_button("Download Features File", f, file_name="features.pkl")

if menu == "Logistic Regression":
    if not st.session_state["logged_in"]:
        st.warning("🔒 Please login first.")
    else:
        st.title("📈 Logistic Regression")
        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please preprocess and upload data first.")
        else:
            df = st.session_state["preprocessed_data"].copy()
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            target_col = st.selectbox("Select Target Variable (Y - binary)", numeric_cols)
            feature_cols = st.multiselect("Select Feature Columns (X)", [col for col in numeric_cols if col != target_col])

            if st.button("Train Logistic Model"):
                X = df[feature_cols]
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("Model Performance:")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                st.write(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
                st.write(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
                st.write(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                joblib.dump(model, "logistic_model.pkl")
                joblib.dump(feature_cols, "features.pkl")
                with open("logistic_model.pkl", "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="logistic_model.pkl")
                with open("features.pkl", "rb") as f:
                    st.download_button("Download Features File", f, file_name="features.pkl")

# ---------------------- PREDICTION ----------------------
if menu == "Prediction":
    if not st.session_state["logged_in"]:
        st.warning("🔒 Please login first.")
    else:
        st.title("🔮 Prediction Interface")
        model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"], key="model")
        features_file = st.file_uploader("Upload Features File (.pkl)", type=["pkl"], key="features")
        if model_file is not None and features_file is not None:
            model = joblib.load(model_file)
            features = joblib.load(features_file)
            st.success("Model and feature files loaded successfully!")
            st.subheader("Enter Input for Prediction:")
            input_data = {}
            for feature in features:
                input_data[feature] = st.number_input(f"Enter value for {feature}")
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                if isinstance(model, LogisticRegression):
                    prob_success = model.predict_proba(input_df)[0][1]
                    if prediction == 1:
                        st.success(f"Predicted Output: Yes (Probability of Success: {prob_success*100:.2f}%)")
                    else:
                        st.success(f"Predicted Output: No (Probability of Failure: {(1-prob_success)*100:.2f}%)")
                else:
                    st.success(f"Predicted Output: {prediction}")
