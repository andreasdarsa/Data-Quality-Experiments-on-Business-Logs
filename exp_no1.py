## FIRST EXPERIMENT SCRIPT

# Given a .csv file with business log data, this script:
# 1. Performs basic preprocessing to ensure we have the following columns:
#    - case_id
#    - activity
#    - timestamp (complete/start)

import pandas as pd
import pm4py
from pm4py.statistics.traces.generic.pandas import case_statistics
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algorithm
import random as rnd
import os

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

def degrade_log(log, degradation_percentage, remove_start_events=False, remove_end_events=False):
    degraded_log = log.copy()
    num_events = len(degraded_log)
    num_to_degrade = int(num_events * degradation_percentage / 100)
    
    if remove_start_events:
        case_ids = degraded_log['case_id'].unique()
        num_cases_to_degrade = int(len(case_ids) * degradation_percentage / 100)
        cases_to_degrade = rnd.sample(list(case_ids), num_cases_to_degrade)
        for case_id in cases_to_degrade:
            degraded_log.loc[degraded_log['case_id'] == case_id, 'start_timestamp'] = pd.NaT
    else:
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

def mine_process_model(log, output_pnml_path):
    # Convert DataFrame to PM4Py event log
    event_log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='start_timestamp')
    
    # Discover process model using Inductive Miner
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
    
    # Export Petri net to PNML file
    pm4py.write_pnml(net, initial_marking, final_marking, output_pnml_path)
    
    # Calculate metrics
    fitness = pm4py.fitness_alignments(event_log, net, initial_marking, final_marking)
    precision = pm4py.precision_alignments(event_log, net, initial_marking, final_marking)

    trace_stats = case_statistics.get_variant_statistics(event_log)
    num_variants = len(trace_stats)
    
    return {
        'fitness': fitness,
        'precision': precision,
        'num_variants': num_variants
    }

# 2. Baseline process mining using PM4Py + inductive miner to discover a process model. It produces:
#    - a Petri net saved as a .pnml file
#    - metrics: fitness, precision, number of sequences/parallel/choices
#    - this forms a baseline of how the process looks like without any advanced techniques

def baseline_experiment():
    input_file = 'data/synthetic_event_log.csv'  # Input CSV file path
    baseline_pnml = 'models/pnml/baseline_model.pnml'
    
    log = preprocess_log(input_file)
    
    # Baseline mining
    baseline_metrics = mine_process_model(log, baseline_pnml)
    print("Baseline Metrics:", baseline_metrics)

# 3. Controlled degradation (damaged logs)
#    - randomly remove start timestamps from a percentage of events
#    - removal of start events from a percentage of cases
#    - partial degradation at 30%,..,90%

def degradation_experiment():
    input_file = 'data/synthetic_event_log.csv'  # Input CSV file path
    degradation_percentages = [30, 50, 70, 90]
    
    log = preprocess_log(input_file)
    
    for degradation_percentage in degradation_percentages:
        degraded_log = degrade_log(log, degradation_percentage, remove_start_events=True, remove_end_events=True)
        degraded_pnml = f'models/pnml/degraded_model_{degradation_percentage}percent.pnml'
        
        degraded_metrics = mine_process_model(degraded_log, degraded_pnml)
        print(f"Degraded Metrics at {degradation_percentage}%:", degraded_metrics)

# 4. Process mining on degraded logs
#    - same as step 2, but on degraded logs
#    - inductive miner sees unclear orderings and creates more parallel constructs/silent transitions
#    - less precision and interpretability, more parallelism

# 5. Comparison and analysis
#    - compare metrics from step 2 and step 4 (baseline and degraded)
#    - process trees and metrics to show impact of degradation
#    - show impact of data quality on process mining results
#    - connection with anomaly detection
#         - pseudo-anomalies (false positives) due to degradation
#         - real anomalies hidden by degradation (false negatives)

def detect_anomalies(log, model_net, initial_marking, final_marking):
    """
    Detects anomalies in the event log using alignment-based conformance checking.
    Returns a list of anomalous case_ids and their alignment costs.
    """
    event_log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='complete_timestamp', start_timestamp_key='start_timestamp')
    alignments = align_algorithm(event_log, model_net, initial_marking, final_marking)
    anomalies = []
    for idx, alignment in alignments.iterrows():
        if alignment['cost'] > 0:
            case_id = alignment['case_id'] if 'case_id' in alignment else event_log.iloc[idx]['case_id']
            anomalies.append({'case_id': case_id, 'cost': alignment['cost']})
    return anomalies

def show_process_model(pnml_path):
    net, initial_marking, final_marking = pm4py.read_pnml(pnml_path)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)
    img_dir = 'models/img'
    os.makedirs(img_dir, exist_ok=True)
    base_name = os.path.basename(pnml_path).replace('.pnml', '.png')
    img_path = os.path.join(img_dir, base_name)
    pn_visualizer.save(gviz, img_path)

if __name__ == "__main__":
    print("Running Baseline Experiment...")
    baseline_experiment()
    
    print("\nRunning Degradation Experiment...")
    degradation_experiment()

    # Example of visualizing a process model
    show_process_model('models/pnml/baseline_model.pnml')
    show_process_model('models/pnml/degraded_model_30percent.pnml')
    show_process_model('models/pnml/degraded_model_50percent.pnml')
    show_process_model('models/pnml/degraded_model_70percent.pnml')
    show_process_model('models/pnml/degraded_model_90percent.pnml')
    # Example of anomaly detection
    input_file = 'data/synthetic_event_log.csv'  # Input CSV file path
    log = preprocess_log(input_file)
    net, im, fm = pm4py.read_pnml('models/pnml/baseline_model.pnml')
    anomalies = detect_anomalies(log, net, im, fm)
    print(f"Detected {len(anomalies)} anomalies:", anomalies)