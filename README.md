# Data Quality Experiments on Business Logs

This repository contains various experiments centered around the impact of data quality issues on process mining

# 1st experiment: Impact of missing timestamps

The first experiment focuses specifically on the impact of missing timestamps on Anomaly Detection reliability
We use a synthetic management flow to compare a clean baseline against logs with varying levels of data damage (% of timestamps removed).

## Process Flow
The experiment uses a 5-step order management process:
1. Receive order
2. Validate Order
3. Approve Order
4. Execute Order
5. Close Order

## Experimental Design
We made use of **1000 cases** generated synthetically

The log contains ~308 real anomalies:
- **Rejections (20%)**: Skipping approval and execution
- **Loops (10%)**: Repeating the Execution step

Afterwards, timestamps were randomly removed at 30%, 50%, 70% and 90% levels to simulate poor data quality

## Key findings
- At 0% degradation, we achieve perfect precision and recall. By 30%, precision collapses to 0.308 as the model becomes too "noisy"
- Even at 90% degradation, fitness remains near 0.999. This proves that fitness is a poor metric for diagnostic reliability.
- Data degradation leads to a false positive explosion, where normal cases are incorrectly flagged as anomalies.

## Repository structure
- ```log_gen.py```: script to generate the synthetic event log.
- ```exp_no1.py```: main experiment script for mining, degradation and alignment
- ```models/pnml/```: contains the petri nets for each stage
- ```models/img/```: contains visualisations of petri nets for each stage
- ```results/```: CSV data and impact plots (bar chart of degradation percentage vs anomalies detected)
