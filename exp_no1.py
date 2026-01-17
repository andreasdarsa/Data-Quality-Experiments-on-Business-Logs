## FIRST EXPERIMENT SCRIPT

# Built-in modules
import random as rnd
import os
import logging
import warnings

# PM4Py modules
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algorithm

# Data handling and plotting modules
import pandas as pd
import matplotlib.pyplot as plt


# Suppress PM4Py warnings for cleaner output (true ballbreakers indeed)
warnings.filterwarnings("ignore")
logging.getLogger("pm4py").setLevel(logging.ERROR)

# Create necessary directories
os.makedirs('models/pnml', exist_ok=True)
os.makedirs('models/img', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Function to preprocess the log (ensure required columns and datetime format)
# - file_path: path to the CSV file
def preprocess_log(file_path):
    log = pd.read_csv(file_path)
    # Ensure required columns exist
    required_columns = ['case_id', 'activity', 'start_timestamp', 'complete_timestamp']
    for col in required_columns:
        if col not in log.columns:
            raise ValueError(f"Missing required column: {col}")
    # Convert timestamp to datetime
    log['start_timestamp'] = pd.to_datetime(log['start_timestamp'])
    log['complete_timestamp'] = pd.to_datetime(log['complete_timestamp'])
    return log

# Function used to degrade the log by removing timestamps
# - degradation_percentage: percentage of events to degrade
# - remove_start_events: if True, remove start timestamps
# - remove_end_events: if True, remove end timestamps
def degrade_log(log, degradation_percentage, remove_start_events=False, remove_end_events=False):
    degraded_log = log.copy()
    num_events = len(degraded_log)
    num_to_degrade = int(num_events * degradation_percentage / 100)
    
    # Degrade start timestamps
    if remove_start_events:
        # Randomly select cases for start timestamp degradation
        case_ids = degraded_log['case_id'].unique()
        num_cases_to_degrade = int(len(case_ids) * degradation_percentage / 100)
        cases_to_degrade = rnd.sample(list(case_ids), num_cases_to_degrade)
        for case_id in cases_to_degrade:
            degraded_log.loc[degraded_log['case_id'] == case_id, 'start_timestamp'] = pd.NaT
    else:
        # Randomly select events for start timestamp degradation
        indices_to_degrade = rnd.sample(range(num_events), num_to_degrade)
        for idx in indices_to_degrade:
            degraded_log.at[idx, 'start_timestamp'] = pd.NaT

    if remove_end_events:
        if remove_start_events:
            # Use the same cases as above for end timestamps
            for case_id in cases_to_degrade:
                degraded_log.loc[degraded_log['case_id'] == case_id, 'complete_timestamp'] = pd.NaT
        else:
            # Randomly select events for end timestamp degradation
            indices_to_degrade_end = rnd.sample(range(num_events), num_to_degrade)
            for idx in indices_to_degrade_end:
                degraded_log.at[idx, 'complete_timestamp'] = pd.NaT

    return degraded_log

# Function to mine a process model from the log using Inductive Miner
# - log: DataFrame with event log
# - output_pnml_path: path to save the mined Petri net in PNML format
# - noise_val: noise threshold for the Inductive Miner
# Returns:
# - net: Petri net model
# - initial_marking: initial marking of the Petri net
# - final_marking: final marking of the Petri net
# - fitness: fitness metric of the model
# - precision: precision metric of the model
def mine_process_model(log, output_pnml_path, noise_val=0.0):
    # Convert DataFrame to PM4Py event log
    event_log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='start_timestamp')
    
    # Discover process model using Inductive Miner
    # Noise threshold allows the model to generalize better with degraded logs
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log, noise_threshold=noise_val)
    
    # Export Petri net to PNML file
    pm4py.write_pnml(net, initial_marking, final_marking, output_pnml_path)
    
    # Calculate metrics
    fitness = pm4py.fitness_alignments(event_log, net, initial_marking, final_marking)
    precision = pm4py.precision_alignments(event_log, net, initial_marking, final_marking)
    
    return net, initial_marking, final_marking, fitness, precision

# Detects anomalies by checking for standard alignment costs OR the use of invisible/skip transitions (moves in model only).
# - log: DataFrame with event log
# - model_net: Petri net model
# - initial_marking: initial marking of the Petri net (placement of tokens at start)
# - final_marking: final marking of the Petri net (placement of tokens at end)
# Returns a list of detected anomalies with case_id, alignment cost, and skips used.
def detect_anomalies(log, model_net, initial_marking, final_marking):
    # Convert DataFrame to PM4Py event log
    event_log_df = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='start_timestamp')
    event_log = pm4py.convert_to_event_log(event_log_df)
    # Compute alignments
    alignments = align_algorithm.apply(event_log, model_net, initial_marking, final_marking)
    
    anomalies = []
    for idx, alignment in enumerate(alignments):
        # An anomaly is detected if there is a non-zero cost or invisible skips
        case_id = event_log[idx].attributes['concept:name']
        # Extract cost and invisible skips
        cost = alignment['cost']
        invisible_skips = sum(1 for move in alignment['alignment'] if move[0] == ">>")
        
        # Record anomaly if cost > 0 or invisible skips > 0
        if cost > 0:
            anomalies.append({
                'case_id': case_id,
                'alignment_cost': cost,
                'skips_used': invisible_skips
            })
    return anomalies

# Function to create and save a visualization of the Petri net
# - pnml_path: path to the PNML file of the Petri net
def create_petri_net_visualization(pnml_path):
    # Create and save a visualization of the Petri net
    net, initial_marking, final_marking = pm4py.read_pnml(pnml_path)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    img_dir = 'models/img'
    os.makedirs(img_dir, exist_ok=True)
    base_name = os.path.basename(pnml_path).replace('.pnml', '.png')
    img_path = os.path.join(img_dir, base_name)
    pn_visualizer.save(gviz, img_path)
    # No visualization (view) in this script

def run_experiments():
    input_file = 'data/synthetic_event_log.csv'  # Input CSV file path
    log = preprocess_log(input_file)
    full_log = log.copy()

    # 1. Identify true anomalies based on predefined rules
    case_activities = full_log.groupby('case_id')['activity'].apply(list).to_dict()
    true_anomalies = set()
    for case_id, activities in case_activities.items():
        # If one of the following conditions is met, mark as true anomaly:
        # Rejection: Approve Order is missing
        # Loop: Execute Order appears more than once
        if 'Approve Order' not in activities or activities.count('Execute Order') > 1:
            true_anomalies.add(str(case_id))

    total_true_count = len(true_anomalies)

    # 2. Experiment with different degradation levels
    results_data = []

    percentages = [0, 30, 50, 70, 90]
    for perc in percentages:
        if perc == 0: # Baseline model
            print(f"\nBaseline Model")
            formatted_log = pm4py.format_dataframe(full_log, case_id='case_id', activity_key='activity', timestamp_key='start_timestamp')
            variants = pm4py.get_variants(formatted_log)
            most_common_variant = max(variants, key=lambda x: variants[x])
            normal_log = pm4py.filter_variants(formatted_log, [most_common_variant])
            net, im, fm, fit, prec = mine_process_model(normal_log, 'models/pnml/baseline_model.pnml')
        else: # Degraded models
            print(f"\nDegradation at {perc}%")
            degraded_log = degrade_log(full_log, perc, remove_start_events=True, remove_end_events=True)
            net, im, fm, fit, prec = mine_process_model(degraded_log, f'models/pnml/degraded_model_{perc}percent.pnml', noise_val=0.2)

        anomalies = detect_anomalies(full_log, net, im, fm)
        detected_ids = set(str(anom['case_id']) for anom in anomalies)

        tp = len(detected_ids.intersection(true_anomalies)) # Real anomalies detected
        fp = len(detected_ids) - tp # Normal cases incorrectly flagged
        fn = total_true_count - tp # Real anomalies missed

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # How many detected anomalies were real
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # How many real anomalies were detected

        results_data.append({
            'degradation': perc,
            'fitness': fit['log_fitness'],
            'precision model': prec,
            'anomalies detected': len(anomalies),
            'true positives': tp,
            'false positives': fp,
            'false negatives': fn,
            'detection precision': precision,
            'detection recall': recall
        })

    # 3. Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('results/experiment_results.csv', index=False)
    print("\nExperiment results saved to 'results/experiment_results.csv'")

    # 4. Display Table
    print("Summary metrics:")
    print(results_df.to_string(index=False))

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['degradation'].astype(str), results_df['anomalies detected'], color='skyblue')
    plt.xlabel('Degradation Percentage')
    plt.ylabel('Number of Anomalies Detected')
    plt.title('Impact of Degradation on Anomaly Detection')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('results/anomalies_vs_degradation.png')
    print("Anomalies vs Degradation plot saved to 'results/anomalies_vs_degradation.png'")
    plt.close()

    # 6. Create visualizations of process models
    create_petri_net_visualization('models/pnml/baseline_model.pnml')
    create_petri_net_visualization('models/pnml/degraded_model_30percent.pnml')
    create_petri_net_visualization('models/pnml/degraded_model_50percent.pnml')
    create_petri_net_visualization('models/pnml/degraded_model_70percent.pnml')
    create_petri_net_visualization('models/pnml/degraded_model_90percent.pnml')

if __name__ == "__main__":
    run_experiments()