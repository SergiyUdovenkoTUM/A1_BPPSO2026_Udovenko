#Imports
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as plt
import pm4py as pm
import sklearn as sk
import statistics as st
#Store data paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
XES_PATH = DATA_DIR / "BPIC17.xes"
CSV_PATH = DATA_DIR / "BPIC17.csv"
#Main
def main():
    #Import data
    if not XES_PATH.is_file():
        raise FileNotFoundError("XES_PATH does not exist")
    print("Reading .xes data from: " + str(XES_PATH))
    data = pm.read_xes(str(XES_PATH))
    print("Data read successfully")

    #Check data structure
    #print("First five events:\n" + data.head(5).to_string())
    #Task 3.2
    #Report basic event log statistics in a table, including:
    #number of cases., number of events., number of process
    #variants., number of distinct case and event attribute
    #labels?, mean. and standard. deviation of case length,
    #mean. and standard deviation. of case duration (in days,
    #minutes, and seconds), number of categorical event
    #attributes.
    #Provide at least two additional statistics of your
    #choice. Here, you can also consider statistics that
    #can be visualized in diagrams.
    case_num = data["case:concept:name"].nunique()
    event_num = len(data)
    print("Number of cases: " + str(case_num))
    print("Number of events: " + str(event_num))
    variants = pm.get_variants(data)
    print("Number of process variants: " + str(len(variants)))
    print("Number of distinct case attribute labels: " + str(len(pm.get_trace_attributes(data))))
    print("Number of distinct event attribute labels: " + str(len(pm.get_event_attributes(data))-len(pm.get_trace_attributes(data))))
    print("Case length mean: " + str(event_num/case_num))
    print("Case length SD: " + str(data.groupby("case:concept:name").size().std(ddof = 0)))
    case_durations = pm.get_all_case_durations(data)
    print("Case duration mean (seconds): " + str(st.mean(case_durations)))
    print("Case duration mean (minutes): " + str((st.mean(case_durations))/60))
    print("Case duration mean (days): " + str((st.mean(case_durations)) / (60*60*24)))
    print("Case duration SD (seconds): " + str(st.stdev(case_durations)))
    print("Case duration SD (minutes): " + str((st.stdev(case_durations)) / 60))
    print("Case duration SD (days): " + str((st.stdev(case_durations)) / (60 * 60 * 24)))
    print("Number of categorical event attributes: 10?") #From manual .csv observation
    print("Median case duration (seconds): " + str(st.median(case_durations))) #Statistic of choice 1
    print("Median case duration (minutes): " + str((st.median(case_durations)) / 60))
    print("Median case duration (days): " + str((st.median(case_durations)) / (60 * 60 * 24)))

    covered_cases = 0
    used_variants = 0
    for i in sorted(variants.values(), reverse = True):
        used_variants += 1
        covered_cases += i
        if covered_cases >= 0.8*case_num:
            break

    print("Minimal number of process variants to cover at least 80% of cases: " + str(used_variants))  # Statistic of choice 2
    print("Actual coverage: " + str(covered_cases/case_num))





#Run
if __name__ == "__main__":
    main()