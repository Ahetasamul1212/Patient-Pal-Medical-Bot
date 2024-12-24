import pandas as pd
import streamlit as st

# Load and preprocess the dataset
data = pd.read_csv(r"E:\MY_python_projects\Medicine Recommendor\archive (2)\medicine.csv")
data.fillna("Not Available", inplace=True)
data['brand name'] = data['brand name'].str.lower()
medicine_lookup = data.set_index('brand name').T.to_dict()

# Query function
def query_medicine_model(user_input, lookup):
    user_input = user_input.lower()
    for token in user_input.split():
        if token in lookup:
            details = lookup[token]
            return (
                f"**Medicine Details for {token.title()}**:\n\n"
                f"- **Dosage Form**: {details.get('dosage form', 'N/A')}\n"
                f"- **Generic**: {details.get('generic', 'N/A')}\n"
                f"- **Strength**: {details.get('strength', 'N/A')}\n"
                f"- **Manufacturer**: {details.get('manufacturer', 'N/A')}\n"
                f"- **Package Info**: {details.get('package container', 'N/A')}"
            )
    return "Sorry, I couldn't find information about that medicine. Please try again."

# Streamlit WebUI
st.title("Medicine Chatbot")
st.write("Welcome! Ask about a medicine to get detailed information.")

# User input
user_input = st.text_input("Type your query here:")

# Display response
if user_input:
    response = query_medicine_model(user_input, medicine_lookup)
    st.markdown(response)
