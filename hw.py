from variable import Variable
from typing import Dict, List
from probGraphicalModels import BeliefNetwork
from probStochSim import RejectionSampling
from probFactors import Prob
from probVE import VE
import json
from os import path
import timeit
import csv
from collections import namedtuple

def perform_exact_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], ordering: List[Variable]) -> Dict[int, float]:
    """ Computes P(Q | E) on a Bayesian Network using variable elimination
        Arguments:
            model, the Bayesian Network
            Q, the query variable
            E, the evidence
            ordering, the order in which variables are eliminated
        
        Returns
            result, a dict mapping each possible value (q) of Q to the probability P(Q = q | E)
    """
    # Use the VE class to perform variable elimination
    ve = VE(gm=model)
    result = ve.query(Q, E, elim_order=ordering)
    return result


def perform_approximate_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], n_samples: int) -> Dict[int, float]:
    """
        Arguments:
            model, the Bayesian Network
            Q, the query variable
            E, the evidence
            n_samples, the number of samples used to approximate P(Q | E)
        
        Returns
            result, a dict mapping each possible value (q) of Q to the probability P(Q = q | E)
    """
    # Use the RejectionSampling class to perform approximate inference
    rj = RejectionSampling(model)
    result = rj.query(qvar=Q, obs=E, number_samples=n_samples)
    return result


# helper function to load some helper variables for use in the main script
def create_helper_variables(dirname: str):
    Helpers = namedtuple("Helpers", ["nodes", "cpts", "name_to_node", "name_to_value", "value_to_name"])

    variables = json.load(open(path.join(dirname, "variables.json")))

    name_to_node = {}
    nodes = []
    value_to_name = {}
    name_to_value = {}
    for variable in variables:
        variable_name = variable["name"]
        variable_values = variable["values"]
        variable_value_names = variable["value_names"]
        node =  Variable(variable_name, variable_values)
        name_to_node[variable_name] = node 
        nodes.append(node)

        value_to_name[variable_name] = { value: name for value, name in zip(variable_values, variable_value_names)}
        name_to_value[variable_name] = { name: value for value, name in zip(variable_values, variable_value_names)}

    tables = json.load(open(path.join(dirname, "tables.json")))
    cpts = []
    for table in tables:
        variable_name = table["variable"]
        node = name_to_node[variable_name]
        parent_names = table["parents"]
        parents = [name_to_node[parent_name] for parent_name in parent_names]
        probability_values = table["values"]
        cpt = Prob(node, parents, probability_values)
        cpts.append(cpt)
    
    return Helpers(nodes, cpts, name_to_node, name_to_value, value_to_name)


def print_head(text: str):
    # helper function to print with overlines and underlines to specify the output
    # if it's printed on console
    print("="*len(text))
    print(text)
    print("="*len(text))

def print_approx_infer_results(input: dict):
    # helper function for printing approx. infer. results
    for disease_value, probability in input.items():
        if disease_value == 'raw_counts':
            pass
        else:
            disease_name = Helpers.value_to_name["Disease"][disease_value]
            print(f"P(Disease = {disease_name}) = {probability:.6f}")

def compute_mse(predicted: Dict[int, float], actual: Dict[int, float]):
    # helper function for computing MSE
    del predicted['raw_counts'] # preprocessing step
    mse = 0.0
    for value in predicted:
        mse += abs(predicted[value] - actual[value]) ** 2
    return mse/len(predicted)


if __name__ == "__main__":
    '''
    ======
    Part 1
    ======
    '''

    '''Question 1'''
    # Construct an instance of the BeliefNetwork class.
    # 1. Load the bayesian network from the directory named "child"

    # create nodes, cpts, name_to_node, name_to_value, value_to_name variables
    Helpers = create_helper_variables("child")

    bn = BeliefNetwork("child", Helpers.nodes, Helpers.cpts)

    '''Question 2 - function implemented above'''

    '''Question 3'''
    # Compute P(Q | E) on the BN using exact inference (variable elimination)
    # to compute  P(Disease | CO2Report = 1, XrayReport = 0, Age = 0) 
    ''' a) Use ascending order of the names of the variables as the elimination ordering '''
    alphabetical_order_nodes = [Helpers.name_to_node[name] for name in sorted([node.__str__() for node in Helpers.nodes])]
    
    Q = Helpers.name_to_node["Disease"]
    E = {Helpers.name_to_node["Age"]: 0, Helpers.name_to_node["CO2Report"]: 1, Helpers.name_to_node["XrayReport"]: 0}

    result_alphabetical_order = perform_exact_inference(model=bn, Q=Q, E=E, ordering=alphabetical_order_nodes)
    print_head("Part 1")
    print_head("Q3a) Result of P(Disease | CO2Report=1, XrayReport=0, Age=0) using Alphabetical Ordering:")
    for disease_value, probability in result_alphabetical_order.items():
        disease_name = Helpers.value_to_name["Disease"][disease_value]
        print(f"P(Disease = {disease_name}) = {probability:.6f}")

    ''' b) Use a better ordering. ''' 
    #   For a better ordering, we can use min-fill ordering i.e. eliminate those 
    #   variables first whose removal would require the least introduction of edges
    #   between the remaining neighbors. Based on observation of the graph, we can set the
    #   min-fill order.

    min_fill_order_nodes = [Helpers.name_to_node[name] for name in [ \
        "BirthAsphyxia",
        "GruntingReport",
        "CO2Report",
        "XrayReport",
        "LVHreport",
        "LowerBodyO2",
        "RUQO2",
        "Age",
        "Grunting",
        "HypDistrib",
        "HypoxiaInO2",
        "CO2",
        "ChestXray",
        "Sick",
        "DuctFlow",
        "CardiacMixing",
        "LungParench",
        "LungFlow",
        "LVH",
        "Disease"
    ]]

    result_minfill_ordering = perform_exact_inference(bn, Q, E, min_fill_order_nodes)

    print_head("Q3b) Result of P(Disease | CO2Report=1, XrayReport=0, Age=0) using Min-fill Ordering:")
    for disease_value, probability in result_alphabetical_order.items():
        disease_name = Helpers.value_to_name["Disease"][disease_value]
        print(f"P(Disease = {disease_name}) = {probability:.6f}")

    '''Question 4'''
    # Time both methods using timeit
    num_runs = 10
    time_alphabetical_order = timeit.timeit(
        lambda: perform_exact_inference(bn, Q, E, alphabetical_order_nodes), 
        number=num_runs
    ) / num_runs

    time_minfill_order = timeit.timeit(
        lambda: perform_exact_inference(bn, Q, E, min_fill_order_nodes), 
        number=num_runs
    ) / num_runs     

    print_head("Q4) Runs for Inference using Variable Elimination through timeit module")
    print(f"Average time taken (alphabetical order): {time_alphabetical_order:.6f} seconds")
    print(f"Average time taken (better order): {time_minfill_order:.6f} seconds")

    # Save results to CSV
    with open("part1.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([time_alphabetical_order, time_minfill_order])


    '''
    ======
    Part 2
    ======
    '''

    '''Question 1 - function implemented above'''

    '''Question 2 - Use the perform_approximate_inference function to compute the same query as part 1'''
    '''a) Case 1: use 10 samples'''
    result_10_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=10)
    print("\n")
    print_head("Part 2")
    print_head("Q2a) Results of Approximate Inference using 10 samples")
    print_approx_infer_results(result_10_samples)

    '''b) Case 2: use 100 samples'''
    result_100_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=100)
    print_head("Q2b) Results of Approximate Inference using 100 samples")
    print_approx_infer_results(result_100_samples)

    '''c) Case 3: Use a number of samples that guarantees PAC(epsilon = 0.01, delta = 0.05)
    Refer https://artint.info/3e/html/ArtInt3e.Ch9.S7.html#SS1.SSSx2
    
    In this case, we will use n_samples = 18445 which corresponds to epsilon = 0.01 and delta = 0.05'''
    result_PAC_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=18445)
    print_head("Q2c) Results of Approximate Inference using 18445 samples (satisfy PAC)")
    print_approx_infer_results(result_PAC_samples)

    '''Question 3'''

    # timing the runs
    num_runs = 10
    time_10_samples = timeit.timeit(
        lambda: perform_approximate_inference(bn, Q, E, 10),
        number = num_runs
    )/num_runs

    time_100_samples = timeit.timeit(
        lambda: perform_approximate_inference(bn, Q, E, 100),
        number = num_runs
    )/num_runs

    time_PAC_samples = timeit.timeit(
        lambda: perform_approximate_inference(bn, Q, E, 18445),
        number = num_runs
    )/num_runs

    print_head("Q3) Runs for Inference using Rejection Sampling through timeit module")
    print(f"Average time taken (10 samples): {time_10_samples:.6f} seconds")
    print(f"Average time taken (100 samples): {time_100_samples:.6f} seconds")
    print(f"Average time taken (18445 samples): {time_PAC_samples:.6f} seconds")

    # Save results to CSV
    with open("part21.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([time_10_samples, time_100_samples, time_PAC_samples])

    '''Question 4'''
    mse_10_samples = mse_100_samples = mse_PAC_samples = 0.0
    
    # measure MSE over 10 runs
    for i in range(10):
        result_10_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=10)
        result_100_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=100)
        result_PAC_samples = perform_approximate_inference(model=bn, Q=Q, E=E, n_samples=18445)

        mse_10_samples += compute_mse(result_10_samples, result_alphabetical_order)
        mse_100_samples += compute_mse(result_100_samples, result_alphabetical_order)
        mse_PAC_samples += compute_mse(result_PAC_samples, result_alphabetical_order)
    
    mse_over_10_runs_10_samples = mse_10_samples/10
    mse_over_10_runs_100_samples = mse_100_samples/10
    mse_over_10_runs_PAC_samples = mse_PAC_samples/10

    print_head("Q4) Mean Squared Error over 10 runs")
    print("MSE for approx. inference w/ 10 sample: ", mse_over_10_runs_10_samples)
    print("MSE for approx. inference w/ 100 sample: ", mse_over_10_runs_100_samples)
    print("MSE for approx. inference w/ 18445 sample: ", mse_over_10_runs_PAC_samples)

    # Save results to CSV
    with open("part22.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([mse_over_10_runs_10_samples, mse_over_10_runs_100_samples, mse_over_10_runs_PAC_samples])

    print_head("END")
    ''' END '''









