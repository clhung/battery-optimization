# Battery Optimization with LLM-Powered Explanation

This personal project explores a simple battery charging/discharging scheduling problem using convex optimization, and leverages large language models (LLMs) to interpret and explain the results.

## Overview
This repository is part of a personal learning effort to:

- Deepen understanding of battery economics and dispatch modeling

- Explore how LLMs can enhance interpretability of time-series optimization

- Combine data science, optimization, and AI reasoning in a real-world energy context

The goal is to schedule a battery system given:

1. Predicted solar generation

2. Predicted load demand

3. Electricity price signals

4. Optional dispatch constraints

The optimization is solved using cvxpy, with the ability to fulfill dispatch requirements and minimize total energy cost.
After solving the problem, an OpenAI LLM is used to summarize and explain the battery behavior in the context of load, solar, and pricing.

## How to Run

### Installation:
```bash
uv venv
source .venv/bin/activate
```

### Running the optimization script:
```bash
python battery_optimization/scheduling_script.py
```

### Launch the interactive dashboard
(Requires OpenAI API key)
```bash
strimlit run battery_optimization/app.py
```

## License
MIT - for personal and educational use.
