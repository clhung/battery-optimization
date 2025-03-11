import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def generate_reasoning_openai(
    data_df: pd.DataFrame,
    results_df: pd.DataFrame,
    openai_api_key: str,
):
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=openai_api_key
    )

    prompt_template = PromptTemplate.from_template(
        """Given the following energy forecast and optimized battery schedule:
    - Solar forecast (kW): {solar}
    - Load forecast (kW): {demand}
    - Electricity price forecast ($/kWh): {price}
    - Optimized battery charge (kW): {charge}
    - Optimized battery discharge (kW): {discharge}

    Explain why the battery schedule behaves the way it does in less
    than 150 words.
    Contrast the counterfactual if there's no battery (solar only system)."""
    )

    prompt = prompt_template.invoke(
        {
            "solar": data_df["solar"].tolist(),
            "demand": data_df["demand"].tolist(),
            "price": data_df["price"].tolist(),
            "charge": results_df["charge"].tolist(),
            "discharge": results_df["discharge"].tolist(),
        }
    )

    response = llm.invoke(prompt)
    return response.content
