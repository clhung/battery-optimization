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
    than 100 words."""
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


def compare_with_no_dispatch_openai(
    data_df: pd.DataFrame,
    results_with_dispatch_df: pd.DataFrame,
    results_no_dispatch_df: pd.DataFrame,
    openai_api_key: str,
) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=openai_api_key
    )

    cost_d = (results_with_dispatch_df["grid"] * data_df["price"]).sum()
    cost_wod = (results_no_dispatch_df["grid"] * data_df["price"]).sum()
    prompt_template = PromptTemplate.from_template(
        """
    Compare the following two battery optimization scenarios with and
    without dispatch event:

    - Optimized battery charge with dispatch event (kW): {charge_d}
    - Optimized battery discharge with dispatch event (kW): {discharge_d}
    - Cost with dispatch event (kW): {cost_d}
    - Optimized battery charge without dispatch event (kW): {charge_wod}
    - Optimized battery discharge without dispatch event (kW): {discharge_wod}
    - Cost without dispatch event (kW): {cost_wod}

    Compare key difference between scenarios with and without dispatch events
    in terms of battery schedule and cost.
    Summarize the comparison in less than 80 words.
    """
    )

    prompt = prompt_template.invoke(
        {
            "charge_d": results_with_dispatch_df["charge"].tolist(),
            "discharge_d": results_with_dispatch_df["discharge"].tolist(),
            "cost_d": cost_d,
            "charge_wod": results_no_dispatch_df["charge"].tolist(),
            "discharge_wod": results_no_dispatch_df["discharge"].tolist(),
            "cost_wod": cost_wod,
        }
    )

    response = llm.invoke(prompt)
    return response.content
