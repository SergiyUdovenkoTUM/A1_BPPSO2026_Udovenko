#Imports
from pathlib import Path
import pm4py as pm
import networkx as nx
import statistics as st

from polars import Null


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
    print("Case length SD: " + str(data.groupby("case:concept:name").size().std(ddof = 1)))
    print("Case duration mean (seconds): " + str(st.mean(case_durations)))
    print("Case duration mean (minutes): " + str((st.mean(case_durations)) / 60))
    print("Case duration mean (days): " + str((st.mean(case_durations)) / (60 * 60 * 24)))
    print("Case duration SD (seconds): " + str(st.stdev(case_durations)))
    print("Case duration SD (minutes): " + str((st.stdev(case_durations)) / 60))
    print("Case duration SD (days): " + str((st.stdev(case_durations)) / (60 * 60 * 24)))
    print("Number of categorical event attributes: 7 + 2 IDs")  # From manual .csv observation
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
def self_made_ext_cardoso(net): #Extended Cardoso metric implementation
    result = 0

    place_to_transition_dict = {}
    for place in net.places:
        destination_list = []
        for out_arc in place.out_arcs:
            destination_list.append(out_arc.target)
        place_to_transition_dict[place] = destination_list

    transition_to_place_set_dict = {}
    for transition in net.transitions:
        destination_set = set()
        for out_arc in transition.out_arcs:
            destination_set.add(out_arc.target)
        transition_to_place_set_dict[transition] = destination_set

    for starting_place in place_to_transition_dict.keys():
        distinct_following_sets_list = []
        for intermediate_transition in place_to_transition_dict[starting_place]:
            if transition_to_place_set_dict[intermediate_transition] not in distinct_following_sets_list:
                distinct_following_sets_list.append(transition_to_place_set_dict[intermediate_transition])
                result += 1

    return result


def self_made_ext_cyclomatic(net, im, fm): #Extended Cyclomatic metric implementation
    reachability_g = pm.convert_to_reachability_graph(net, im, fm)
    reachability_di_g = nx.DiGraph()

    reachability_di_g.add_nodes_from(reachability_g.states)
    for transition in reachability_g.transitions:
        reachability_di_g.add_edge(transition.from_state, transition.to_state)

    states_n = reachability_di_g.number_of_nodes()
    edges_n = reachability_di_g.number_of_edges()
    scc_n = nx.number_strongly_connected_components(reachability_di_g)


    return edges_n-states_n+scc_n


#Create, view and evaluate a model
def create_view_evaluate_model(data, f, approach, nt_i = 0.2, dt_h=0.5, at_h=0.65, ltt_h=0.5):
    #Create and View
    match approach:
        case "IM":
            model = pm.discover_petri_net_inductive(data)
            pm.view_petri_net(model[0], model[1], model[2], format=f, graph_title="Inductive Miner")
            print("Inductive Miner:")
        case "IMf":
            model = pm.discover_petri_net_inductive(data, noise_threshold=nt_i)
            pm.view_petri_net(model[0], model[1], model[2], format=f,
                              graph_title="Inductive Miner: noise threshold = " + str(nt_i))
            print("Inductive Miner: noise threshold = " + str(nt_i))
        case "HM":
            model = pm.discover_petri_net_heuristics(data, dependency_threshold=dt_h, and_threshold=at_h, loop_two_threshold=ltt_h)
            pm.view_petri_net(model[0], model[1], model[2], format=f,
                              graph_title="Heuristic Miner: dependency threshold = " + str(
                                  dt_h) + ", and threshold = " + str(at_h) + ", loop two threshold = " + str(ltt_h))
            print("Heuristic Miner: dependency threshold = " + str(dt_h) + ", and threshold = " + str(
                at_h) + ", loop two threshold = " + str(ltt_h))
            is_sound = pm.algo.analysis.woflan.algorithm.apply(model[0], model[1], model[2], parameters={
                "return_asap_when_not_sound": True, "print_diagnostics": True, "return_diagnostics": False})
            print("Is Sound: " + str(is_sound))
        case _:
            print("No such model type: " + str(approach))
            return Null
    #Evaluate
    print("Fitness: " + str(pm.fitness_token_based_replay(data, model[0], model[1], model[2])))
    print("Precision: " + str(pm.precision_token_based_replay(data, model[0], model[1], model[2])))
    print("Generalization: " + str(pm.generalization_tbr(data, model[0], model[1], model[2])))
    print("Extended Cardoso Simplicity Score: " + str(self_made_ext_cardoso(model[0])))
    if approach != "HM": #Due to unsoundness of HM
        print("Extended Cyclomatic Simplicity Score: " + str(self_made_ext_cyclomatic(model[0], model[1], model[2])))

    return model


#Run
if __name__ == "__main__":
    # Store data paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    XES_PATH = DATA_DIR / "BPIC17.xes"
    RESULT_DIR = PROJECT_ROOT / "result"
    BPMN_PATH = RESULT_DIR / "imf_model.bpmn"
    # Import data
    if not XES_PATH.is_file():
        raise FileNotFoundError("XES_PATH does not exist")
    print("Reading .xes data from: " + str(XES_PATH))
    bpic17_data = pm.read_xes(str(XES_PATH))
    print("Data read successfully")


    basic_analysis(bpic17_data) #Task 3.2


    #Task 3.3
    render = "html" #Change for other display options

    #Unsuccessful model tests (percentage_of_fitting_traces<<80 or >>80)
    #create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.2)
    #create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.1)
    #create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.05)
    #create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.025)
    #create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.0275)

    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.5, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.3, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.1, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.7, at_h=0.65, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.5, at_h=0.4, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.5, at_h=0.9, ltt_h=0.5)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.5, at_h=0.65, ltt_h=0.3)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.5, at_h=0.65, ltt_h=0.8)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.1, at_h=0.1, ltt_h=0.1)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.01, at_h=0.01, ltt_h=0.01)
    #create_view_evaluate_model(bpic17_data, f=render, approach="HM", dt_h=0.9, at_h=0.9, ltt_h=0.9)


    #Final models
    print("Working on Inductive Miner")
    create_view_evaluate_model(bpic17_data, f = render, approach="IM") #Reference point
    print("Working on IMf")
    model = create_view_evaluate_model(bpic17_data, f=render, approach="IMf", nt_i = 0.028) #Successful IMf
    print("Working on Heuristic Miner")
    create_view_evaluate_model(bpic17_data, f=render, approach="HM") #Basic Heuristic Miner

    #Final result
    result = pm.convert_to_bpmn(*model)
    pm.write_bpmn(result, file_path=str(BPMN_PATH))
