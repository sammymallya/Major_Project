"""
Evaluation Dashboard using Streamlit.

Displays evaluation results in tables and graphs.
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Evaluation Dashboard")

# Load results
with open("evaluation_results.json", "r") as f:
    data = json.load(f)

queries = data["queries"]

# Prepare data for display
mode_data = {"none": [], "vectordb": [], "kg": [], "hybrid": []}

for query in queries:
    for mode in mode_data.keys():
        if mode in query["modes"] and "error" not in query["modes"][mode]:
            metrics = query["modes"][mode]
            mode_data[mode].append({
                "query": query["query"][:50] + "...",
                "semantic_score": metrics["semantic_score"],
                "f1_score": metrics["f1_score"],
                "hallucination_rate": metrics["hallucination_rate"],
                "improvement_semantic": metrics.get("improvement_over_none", {}).get("semantic"),
                "improvement_f1": metrics.get("improvement_over_none", {}).get("f1"),
                "improvement_hallucination": metrics.get("improvement_over_none", {}).get("hallucination"),
            })

# Display tables
st.header("Metrics per Mode")
for mode, records in mode_data.items():
    if records:
        df = pd.DataFrame(records)
        st.subheader(f"Mode: {mode}")
        st.dataframe(df)

# Average metrics
st.header("Average Metrics")
avg_data = {}
for mode, records in mode_data.items():
    if records:
        df = pd.DataFrame(records)
        avg_data[mode] = {
            "avg_semantic": df["semantic_score"].mean(),
            "avg_f1": df["f1_score"].mean(),
            "avg_hallucination": df["hallucination_rate"].mean(),
        }

avg_df = pd.DataFrame.from_dict(avg_data, orient="index")
st.dataframe(avg_df)

# Improvements over none
st.header("Improvements over 'none' Mode")
improvement_data = {}
for mode in ["vectordb", "kg", "hybrid"]:
    records = mode_data[mode]
    if records:
        df = pd.DataFrame(records)
        improvement_data[mode] = {
            "avg_improvement_semantic": df["improvement_semantic"].mean(),
            "avg_improvement_f1": df["improvement_f1"].mean(),
            "avg_improvement_hallucination": df["improvement_hallucination"].mean(),
        }

if improvement_data:
    imp_df = pd.DataFrame.from_dict(improvement_data, orient="index")
    st.dataframe(imp_df)

    # Bar chart for improvements
    fig, ax = plt.subplots()
    imp_df.plot(kind="bar", ax=ax)
    st.pyplot(fig)