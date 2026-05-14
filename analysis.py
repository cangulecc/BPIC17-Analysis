import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics

random.seed(42)
np.random.seed(42)

print("Loading event log")
log_path = "data.xes"
log = xes_importer.apply(log_path)
df = pm4py.convert_to_dataframe(log)

# 1. Basic numbers
num_cases = len(log)
num_events = len(df)
variants = variants_filter.get_variants(log)
num_variants = len(variants)

#2. Attribute numbers
case_attributes = pm4py.get_trace_attributes(log)
num_case_attributes = len(case_attributes)

num_event_attributes = len(df.columns)
num_categorical_attributes = len(df.select_dtypes(include=['object','category']).columns)

#3. Case lengths (How many activities are there?)
case_lengths = [len(trace) for trace in log]
mean_length = np.mean(case_lengths)
std_length = np.std(case_lengths)

#4. Case durations
all_durations = case_statistics.get_all_case_durations(log)
mean_dur_sec = np.mean(all_durations)
std_dur_sec = np.std(all_durations)

#Function to format durations (days, minutes, seconds),
def format_duration(seconds):
    days = seconds / 86400
    minutes = seconds / 60
    return f"{days:.2f} d, {minutes:.2f} min, {seconds:.2f} sec"

#Results
print(f"\n--- Basic Statistics ---")
print(f"Number of cases: {num_cases}")
print(f"Number of events: {num_events}")
print(f"Number of variants: {num_variants}")
print(f"Number of case attributes: {num_case_attributes}")
print(f"Number of event attributes: {num_event_attributes}")
print(f"Number of categorical attributes: {num_categorical_attributes}")
print(f"Mean case length: {mean_length:.2f}")
print(f"Case length standard deviation: {std_length:.2f}")
print(f"Mean case duration: {format_duration(mean_dur_sec)}")
print(f"Case duration standard deviation: {format_duration(std_dur_sec)}")

#Additional statistics and visualizations
#Case arrival rate weekly
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
plt.figure(figsize=(10,4))
df.set_index('time:timestamp').resample('W')['case:concept:name'].nunique().plot()
plt.title("Weekly Case Arrival Rate")
plt.ylabel("New Cases")
plt.savefig("arrival_rate.png")
plt.close()

#Top 10 most common activities
plt.figure(figsize=(10, 4))
df['concept:name'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Activities")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("top_activities.png")
plt.close()

#Process Discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

#Filter 80% of the most common variants (Reduce noise, focus on main behavior)
filtered_log = variants_filter.filter_log_variants_percentage(log, percentage=0.8)
#Simplicity metrics
def calculate_node_count(net):
    "Metric 1: Total number of nodes (The less, the simpler)"
    return len(net.places) + len(net.transitions)

def calculate_arc_per_node(net):
    "Metric 2: Average arcs per node (Link density)"
    nodes = len(net.places) + len(net.transitions)
    arcs = len(net.arcs)
    return arcs / nodes if nodes > 0 else 0

#Discovery
print("Discovering models")
#Algorithm 1: Inductive Miner
tree_ind = pm4py.discover_process_tree_inductive(filtered_log, noise_threshold=0.2)
net_ind, im_ind, fm_ind = pm4py.convert_to_petri_net(tree_ind)

#Algorithm 2: Heuristic Miner
net_heu, im_heu, fm_heu = heuristics_miner.apply(filtered_log, 
    parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.9995,
        heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.90,
        heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 10,
        heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH: 0.90
    })
#Quality metrics
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

def evaluate_model(model_net, initial_marking, final_marking, log_sample):
    print("Evaluating model with Token replay")
    fitness_res = fitness_evaluator.apply(log_sample, model_net, initial_marking, final_marking,
                                          variant=fitness_evaluator.Variants.TOKEN_BASED)
    fitness = fitness_res['log_fitness']

    precision = precision_evaluator.apply(log_sample, model_net, initial_marking, final_marking,
                                          variant = precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    
    generalization = generalization_evaluator.apply(log_sample, model_net, initial_marking, final_marking)
    
    nodes = calculate_node_count(model_net)
    arc_density = calculate_arc_per_node(model_net)
    return {
        "Fitness": fitness,
        "Precision": precision,
        "Generalization": generalization,
        "Node Count (Custom Metric 1)": nodes,
        "Arc Density (Custom Metric 2)": arc_density
    }
#Sample 200 cases for eval
sample_log = pm4py.objects.log.obj.EventLog(random.sample(list(filtered_log), 400))
print("\nResults for Inductive Miner:")
results_ind = evaluate_model(net_ind, im_ind, fm_ind, sample_log)
print(results_ind)
print("\nResults for Heuristic Miner:")
results_heu = evaluate_model(net_heu, im_heu, fm_heu, sample_log)
print(results_heu)

#Save models as BPMN
bpmn_ind = pm4py.convert_to_bpmn(net_ind, im_ind, fm_ind)
pm4py.write_bpmn(bpmn_ind, "model_inductive.bpmn")
bpmn_heu = pm4py.convert_to_bpmn(net_heu, im_heu, fm_heu)
pm4py.write_bpmn(bpmn_heu, "model_heuristic.bpmn")
print("Done! Models saved as BPMN files.")