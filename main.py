#Imports
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as plt
import pm4py as pm
import sklearn as sk
import statistics as st



#Task 3.2
#Calculates and prints the basic metrics for the provided dataset
def basic_analysis(data):
    case_num = data["case:concept:name"].nunique()
    event_num = len(data)
    variants = pm.get_variants(data)
    case_durations = pm.get_all_case_durations(data)
    print("Number of cases: " + str(case_num))
    print("Number of events: " + str(event_num))
    print("Number of process variants: " + str(len(variants)))
    print("Number of distinct case attribute labels: " + str(len(pm.get_trace_attributes(data))))
    print("Number of distinct event attribute labels: " + str(len(pm.get_event_attributes(data)) - len(pm.get_trace_attributes(data))))
    print("Case length mean: " + str(event_num / case_num))
    print("Case length SD: " + str(data.groupby("case:concept:name").size().std(ddof=0)))
    print("Case duration mean (seconds): " + str(st.mean(case_durations)))
    print("Case duration mean (minutes): " + str((st.mean(case_durations)) / 60))
    print("Case duration mean (days): " + str((st.mean(case_durations)) / (60 * 60 * 24)))
    print("Case duration SD (seconds): " + str(st.stdev(case_durations)))
    print("Case duration SD (minutes): " + str((st.stdev(case_durations)) / 60))
    print("Case duration SD (days): " + str((st.stdev(case_durations)) / (60 * 60 * 24)))
    print("Number of categorical event attributes: 10?")  # From manual .csv observation
    print("Median case duration (seconds): " + str(st.median(case_durations)))  # Statistic of choice 1
    print("Median case duration (minutes): " + str((st.median(case_durations)) / 60))
    print("Median case duration (days): " + str((st.median(case_durations)) / (60 * 60 * 24)))

    covered_cases = 0
    used_variants = 0
    for i in sorted(variants.values(), reverse=True):
        used_variants += 1
        covered_cases += i
        if covered_cases >= 0.8 * case_num:
            break

    print("Minimal number of process variants to cover at least 80% of cases: " + str(used_variants))  # Statistic of choice 2
    print("Actual coverage: " + str(covered_cases / case_num))


#Task 3.3
def create_view_evaluate_inductive(data, f):
    #Create
    inductive = pm.discover_petri_net_inductive(data)
    #View
    pm.view_petri_net(inductive[0], inductive[1], inductive[2], format=f, graph_title="Inductive Miner")
    #Evaluate
    print("Inductive Miner")
    print("Fitness: " + str(pm.fitness_token_based_replay(data, inductive[0], inductive[1], inductive[2])))
    #print("Precision: " + str(pm.precision_token_based_replay(data, inductive[0], inductive[1], inductive[2])))
    print("Generalization: " + str(pm.generalization_tbr(data, inductive[0], inductive[1], inductive[2])))
    #print("Is Sound: " + str(pm.algo.analysis.woflan.algorithm.apply(inductive[0], inductive[1], inductive[2],
                                                                     #parameters={"return_asap_when_not_sound": True, "print_diagnostics": True, "return_diagnostics": False})))

def create_view_evaluate_inductive_f(data, f, nt_i = 0.2):
    #Create
    inductive_filtered = pm.discover_petri_net_inductive(data, noise_threshold=nt_i)
    #View
    pm.view_petri_net(inductive_filtered[0], inductive_filtered[1], inductive_filtered[2], format=f, graph_title="Inductive Miner: noise threshold = " + str(nt_i))
    #Evaluate
    print("Inductive Miner: noise threshold = " + str(nt_i))
    print("Fitness: " + str(pm.fitness_token_based_replay(data, inductive_filtered[0], inductive_filtered[1], inductive_filtered[2])))
    #print("Precision: " + str(pm.precision_token_based_replay(data, inductive_filtered[0], inductive_filtered[1], inductive_filtered[2])))
    print("Generalization: " + str(pm.generalization_tbr(data, inductive_filtered[0], inductive_filtered[1], inductive_filtered[2])))
    #print("Is Sound: " + str(pm.algo.analysis.woflan.algorithm.apply(inductive_filtered[0], inductive_filtered[1], inductive_filtered[2],
                                                                     #parameters={"return_asap_when_not_sound": True, "print_diagnostics": True, "return_diagnostics": False})))

def create_view_evaluate_heuristic(data, f, dt_h=0.5, at_h=0.65, ltt_h=0.5):
    heuristic = pm.discover_petri_net_heuristics(data, dependency_threshold=dt_h, and_threshold=at_h,
                                                 loop_two_threshold=ltt_h)
    pm.view_petri_net(heuristic[0], heuristic[1], heuristic[2], format=f,
                      graph_title="Heuristic Miner: dependency threshold = " + str(
                          dt_h) + ", and threshold = " + str(at_h) + ", loop two threshold = " + str(ltt_h))
    print("Heuristic Miner: dependency threshold = " + str(dt_h) + ", and threshold = " + str(
        at_h) + ", loop two threshold = " + str(ltt_h))
    print("Fitness: " + str(pm.fitness_token_based_replay(data, heuristic[0], heuristic[1], heuristic[2])))
    #print("Precision: " + str(pm.precision_token_based_replay(data, heuristic[0], heuristic[1], heuristic[2])))
    print("Generalization: " + str(pm.generalization_tbr(data, heuristic[0], heuristic[1], heuristic[2])))
    #print("Is Sound: " + str(pm.algo.analysis.woflan.algorithm.apply(heuristic[0], heuristic[1], heuristic[2],
                                                                     #parameters={"return_asap_when_not_sound": True, "print_diagnostics": True, "return_diagnostics": False})))




#Run
if __name__ == "__main__":
    # Store data paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    XES_PATH = DATA_DIR / "BPIC17.xes"
    CSV_PATH = DATA_DIR / "BPIC17.csv"
    # Import data
    if not XES_PATH.is_file():
        raise FileNotFoundError("XES_PATH does not exist")
    print("Reading .xes data from: " + str(XES_PATH))
    bpic17_data = pm.read_xes(str(XES_PATH))
    print("Data read successfully")


    #basic_analysis(bpic17_data) #Task 3.2


    #Task 3.3
    render = "html" #Change for other display options


    #Unsuccessful model tests (percentage_of_fitting_traces<<80 or >>80)
    #create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.2)
    #create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.1)
    #create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.05)
    #create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.025)
    #create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.0275)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.5, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.3, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.1, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.7, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.5, at_h=0.4, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.5, at_h=0.9, ltt_h=0.5)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.5, at_h=0.65, ltt_h=0.3)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.5, at_h=0.65, ltt_h=0.8)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.1, at_h=0.1, ltt_h=0.1)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.01, at_h=0.01, ltt_h=0.01)
    #create_view_evaluate_heuristic(bpic17_data, f=render, dt_h=0.9, at_h=0.9, ltt_h=0.9)


    #Final models
    print("Working on Inductive Miner")
    create_view_evaluate_inductive(bpic17_data, f = render) #Reference point
    print("Working on IMf")
    create_view_evaluate_inductive_f(bpic17_data, f=render, nt_i = 0.028) #Successful IMf
    print("Working on Heuristic Miner")
    create_view_evaluate_heuristic(bpic17_data, f=render) #Basic Heuristic Miner