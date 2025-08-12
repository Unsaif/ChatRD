from pathlib import Path
import networkx as nx
import obonet
import pandas as pd
import numpy as np

def load_ontology(annotations='OMIM|ORPHA'):
    # Use pathlib to construct the file paths
    base_path = Path(__file__).parent / '../files'
    hp_obo_path = base_path / 'hp.obo'
    hpoa_path = base_path / 'phenotype.hpoa'

    # Load the ontology
    graph = obonet.read_obo(hp_obo_path)

    # Load the annotations
    df_annotations = pd.read_csv(hpoa_path, sep='\t', comment='#', low_memory=False)
    df_annotations = df_annotations[df_annotations['database_id'].str.contains(annotations)]
    
    # Preprocess the annotations
    df_annotations = df_annotations[['database_id', 'hpo_id']].dropna()

    # Count the number of diseases
    num_diseases = df_annotations['database_id'].nunique()

    # Calculate the frequency of each HPO term
    hpo_counts = df_annotations['hpo_id'].value_counts()

    # Calculate probability of each HPO term
    hpo_probs = hpo_counts / num_diseases

    # Compute IC for each HPO term
    hpo_ic = -np.log(hpo_probs)

    # Initialize counts for all nodes
    ic_dict = {term: 0 for term in graph.nodes()}

    # Set initial counts from hpo_counts
    for hpo_id, count in hpo_counts.items():
        ic_dict[hpo_id] = count

    # Propagate counts up the ontology
    for node in nx.topological_sort(graph):
        for parent in graph.successors(node):
            ic_dict[parent] += ic_dict[node]

    # Recalculate probabilities and ICs after propagation
    total_count = ic_dict['HP:0000001']  # 'All' root term
    for term in ic_dict:
        prob = ic_dict[term] / total_count
        ic_dict[term] = -np.log(prob) if prob > 0 else 0

    return graph, ic_dict

def resnik_similarity(hpo1, hpo2, graph, ic_dict):
    # Get ancestors of each term (including the term itself)
    ancestors1 = nx.ancestors(graph, hpo1) | {hpo1}
    ancestors2 = nx.ancestors(graph, hpo2) | {hpo2}
    # Find common ancestors
    common_ancestors = ancestors1 & ancestors2
    if not common_ancestors:
        return 0
    # Get ICs of common ancestors
    ic_values = [ic_dict[term] for term in common_ancestors]
    # Resnik similarity is the maximum IC among common ancestors
    return max(ic_values)

def groupwise_resnik_similarity(patient_terms, disease_terms, graph, ic_dict):
    similarities = []
    for p_term in patient_terms:
        max_sim = 0
        for d_term in disease_terms:
            sim = resnik_similarity(p_term, d_term, graph, ic_dict)
            if sim > max_sim:
                max_sim = sim
        similarities.append(max_sim)
    # Aggregate similarities (e.g., average)
    if similarities:
        return sum(similarities) / len(similarities)
    else:
        return 0
    
def sim_gic(patient_terms, disease_terms, ic_dict, graph):

    ancestors_p = set()
    for term in patient_terms:
        if term in graph:
            ancestors_p.update(nx.ancestors(graph, term) | {term})
    ancestors_d = set()
    for term in disease_terms:
        if term in graph:
            ancestors_d.update(nx.ancestors(graph, term) | {term})
    intersection = ancestors_p & ancestors_d
    union = ancestors_p | ancestors_d
    numerator = sum([ic_dict[term] for term in intersection])
    denominator = sum([ic_dict[term] for term in union])
    return numerator / denominator if denominator > 0 else 0