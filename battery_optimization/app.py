import streamlit as st
from llm_reasoning import generate_reasoning_openai
from scheduling_script import optimization_scheduler

openai_api_key = st.secrets["OPENAI_API_KEY"]
st.title("🔋Battery Dispatch Reasoning Dashboard")

# Display data
data_df, results_df = optimization_scheduler("2023-01-15", 36)
combined_df = data_df.merge(results_df, on="timestamp")
st.line_chart(
    combined_df,
    x="timestamp",
    y=["solar", "demand", "price", "charge", "discharge"],
)

# Generate reasoning
if st.button("🧠 Explain Battery Behavior"):
    with st.spinner("Analyzing battery schedule..."):
        reasoning = generate_reasoning_openai(
            data_df, results_df, openai_api_key=openai_api_key
        )
    st.subheader("🤖 AI Explanation")
    st.write(reasoning)
