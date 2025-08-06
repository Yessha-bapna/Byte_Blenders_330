
import streamlit as st
import requests
import json
from io import BytesIO

# Set wide layout and apply custom CSS for dark theme and neon blue buttons
st.set_page_config(page_title="HackRx Policy QA", layout="wide")

st.markdown("""
    <style>
        /* Dark background */
        .stApp {
            background-color: #0e0e0e;
            color: white;
        }

        /* Electric Blue Buttons (HEX: #007bff) */
        div.stButton > button {
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.7em 1.5em;
            font-weight: 600;
            font-size: 16px;
            width: 100%; /* Make the button long */
            
        }

        div.stButton > button:hover {
            background-color: white;
            color: #007bff;
            border: 1px solid #007bff;
            box-shadow: 0 0 20px #007bff;
        }

        /* Input box styling */
        .stTextInput input {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #007bff;
        }

        /* File uploader text color */
        .stFileUploader label {
            color: white;
        }

        /* Code block color */
        pre, code {
            color: #007bff;
        }
            
       
    </style>
""", unsafe_allow_html=True)


st.title("📄Intelligent Policy Document Q&A")
st.write("Ask questions from your uploaded insurance PDFs and get accurate, clause-level answers.")

# Divide layout into 2 columns: Upload on left, Q&A on right
col1, col2 = st.columns(2)

# --- Upload PDF Section ---
with col1:
    st.subheader("1️⃣ Upload a Policy Document")
    uploaded_file = st.file_uploader("Upload a .pdf policy document", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Uploading and indexing document..."):
            res = requests.post("http://localhost:8000/upload", files={"file": uploaded_file})
            if res.status_code == 200:
                session_id = res.json().get("session_id")
                st.success("Uploaded successfully!")
                st.session_state["session_id"] = session_id
            else:
                st.error("Upload failed.")

# --- Ask Question Section ---
with col2:
    st.subheader("2️⃣ Ask a Question")
    question = st.text_input("Type your question here")

    if st.button("Ask"):
        if "session_id" not in st.session_state:
            st.warning("Please upload a document first.")
        else:
            with st.spinner("Thinking..."):
                payload = {
                    "session_id": st.session_state["session_id"],
                    "question": question
                }
                res = requests.post("http://localhost:8000/ask_uploaded", json=payload)
                if res.status_code == 200:
                    response = res.json()
                    st.markdown(f"### ✅ Answer: {response.get('answer')}")
                    st.markdown("**Matched Clause:**")
                    st.code(response.get("matched_clause")[:500])
                    st.markdown(f"**Document:** {response.get('document')}")
                    
                    # Add download button
                    json_data = json.dumps(response, indent=4)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="response.json",
                        mime="application/json"
                    )
                else:
                    st.error("Something went wrong. Please check backend logs.")
