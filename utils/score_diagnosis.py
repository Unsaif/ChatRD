from tqdm.auto import tqdm
import numpy as np

from utils.synthetic_dataset_utils import retrieve_nearest_neighbours
import time
import importlib
import utils.overlap_method
importlib.reload(utils.overlap_method)
import json
import pandas as pd

from utils.overlap_method import predict_disease_from_hpo
from utils.hpo_ontology import load_ontology
from utils.hdc import encode_patient_profile, rank_diseases

# Load the HPO ontology
graph_orpha, ic_dict_orpha = load_ontology(annotations='ORPHA')

def score_diagnosis(adapter, predicted_id, gold_standard_id, depth_cache):
    """
    Score a predicted diagnosis using the Mondo ontology.

    Args:
        adapter: Oaklib adapter for Mondo ontology.
        predicted_id: Predicted diagnosis ID (Mondo ID).
        gold_standard_id: Gold standard diagnosis ID (Mondo ID).

    Returns:
        str: "Exact match", "Deepest ancestor match", or "No match".
    """

    # Handle sssom_mappings safely
    results = adapter.sssom_mappings(predicted_id)
    first_mapping = next(results, None)
    if first_mapping:
        predicted_id = first_mapping.subject_id

    if predicted_id == gold_standard_id:
        return "Exact match"

    # Get ancestors for both predicted and gold standard IDs
    predicted_ancestors = {ancestor for ancestor in adapter.ancestors(predicted_id) if ancestor.startswith("MONDO:")}
    gold_standard_ancestors = {ancestor for ancestor in adapter.ancestors(gold_standard_id) if ancestor.startswith("MONDO:")}

    # Calculate current depth for the predicted_id
    current_depth_predicted = depth_cache[predicted_id]
    current_depth_gold = depth_cache[gold_standard_id]

    # Define a function to measure how "close" an ancestor is to predicted_id
    def distance_from_predicted(term, current_depth):
        return current_depth - depth_cache[term]

    non_self_ancestors_predicted = [
        ancestor for ancestor in predicted_ancestors if ancestor != predicted_id
    ]
    non_self_ancestors_gold = [
        ancestor for ancestor in gold_standard_ancestors if ancestor != gold_standard_id
    ]

    if non_self_ancestors_predicted and non_self_ancestors_gold:
        # Get the closest ancestor overall
        nearest_ancestor_predicted = min(non_self_ancestors_predicted, key=lambda term: distance_from_predicted(term, current_depth_predicted))
        nearest_ancestor_gold = min(non_self_ancestors_gold, key=lambda term: distance_from_predicted(term, current_depth_gold))

        if nearest_ancestor_predicted == nearest_ancestor_gold:
            return "Deepest ancestor match"
        else:
            return "No match"
    else:
        return "No match"

    # # Find common ancestors
    # common_ancestors = predicted_ancestors.intersection(gold_standard_ancestors)

    # # Check for exact match
    # if predicted_id == gold_standard_id:
    #     return "Exact match"
    
    # if list(gold_standard_ancestors)[0] in common_ancestors:
    #     # print(f'Gold standard deepest ancestor: {adapter.label(gold_standard_deepest[0])}')
    #     common_ancestors_translated = []
    #     for ancestor in list(common_ancestors):
    #         common_ancestors_translated.append(adapter.label(ancestor))
    #     # print(f'Common ancestors: {common_ancestors_translated}')
    #     return "Deepest ancestor match"
    # else:
    #     return "No match"

def evaluate_predictions_and_save_logs(
    relevant_hpos, prompts_df, adapter, graph, 
    ic_dict, depth_cache, orpha=False, negative_hpos=None, index=None, 
    id_to_disease=None, collection=None, diseases_to_consider=None, semantic_similarity=False, 
    weighted_score_active=False, hnsw=False, hdc=False, hpo_vectors=None, disease_vectors=None, 
    disease_names=None, k=1000, log_file_path="output_log.txt", output_ranks_path="ranks.npz", 
    output_predictions_path="all_predictions.csv"
):
    """
    Function to evaluate predictions, save ranks, and log outputs to a file.

    Args:
        relevant_hpos (dict): Dictionary of HPO terms for each case.
        prompts_df (DataFrame): DataFrame containing the ground truth (correct diagnosis).
        adapter: Adapter object for handling SSSOM mappings and scoring.
        graph: Ontology graph.
        ic_dict (dict): Information content dictionary for HPO terms.
        log_file_path (str): Path to save the log output.
        output_ranks_path (str): Path to save the rank results.
    """
    exact_match_ranks = []
    exact_match_deepest_ancestor_match_ranks = []
    no_of_diseases_considered = []
    exact_match_ranks_dict = {}
    exact_match_deepest_ancestor_match_ranks_dict = {}
    predictions_rows = []

    # Open a log file for writing
    with open(log_file_path, "w") as log_file:
        for key in tqdm(prompts_df.index):
            correct_diagnosis = prompts_df.loc[key, 'OMIM']
            correct_diagnosis = f"OMIM:{correct_diagnosis}"

            if diseases_to_consider is not None:
                correct_disease = prompts_df.loc[key, 'Correct Diagnosis']
                try:
                    if correct_disease.strip() not in diseases_to_consider:
                        continue
                except Exception as e:
                    log_file.write(f"Error fetching correct diagnosis: {e}\n")
                    continue
            
            try:
                relevant_hpos[key]
            except KeyError:
                log_file.write(f"\nFile: {key}\n")
                log_file.write(f"Error fetching HPO terms\n")
                continue

            # Fetch the gold standard ID
            results = adapter.sssom_mappings(correct_diagnosis)
            gold_standard_id = next(results, None)
            gold_standard_id = gold_standard_id.subject_id if gold_standard_id else None

            # Predict diseases
            if hdc:
                try:
                    patient_vector = encode_patient_profile(relevant_hpos[key], hpo_vectors)
                    pot_diseases = rank_diseases(patient_vector, disease_vectors, disease_names)
                    pot_diseases = [x[1] for x in pot_diseases]
                except Exception as e:
                    log_file.write(f'Error retrieving hdc similarity for {key}: {e}')
                    continue
                predicted_diseases = predict_disease_from_hpo(relevant_hpos[key], graph, ic_dict, semantic_similarity=semantic_similarity, pot_diseases=pot_diseases, weighted_score_active=weighted_score_active, neg_hpos=negative_hpos[key] if negative_hpos is not None else None)
            elif hnsw:
                try:
                    pot_diseases = retrieve_nearest_neighbours(relevant_hpos[key], collection, index, id_to_disease, k=k)
                except Exception as e:
                    log_file.write(f'Error retrieving nearest neighbours for {key}: {e}')
                    continue
                predicted_diseases = predict_disease_from_hpo(relevant_hpos[key], graph, ic_dict, semantic_similarity=semantic_similarity, pot_diseases=pot_diseases, weighted_score_active=weighted_score_active, neg_hpos=negative_hpos[key] if negative_hpos is not None else None)
                if orpha:
                    predicted_diseases_orpha = predict_disease_from_hpo(relevant_hpos[key], graph_orpha, ic_dict_orpha, OMIM_or_ORPHA='ORPHA', semantic_similarity=semantic_similarity, weighted_score_active=weighted_score_active, neg_hpos=negative_hpos[key] if negative_hpos is not None else None)
                    predicted_diseases = pd.concat([predicted_diseases, predicted_diseases_orpha])
                    predicted_diseases = predicted_diseases.sort_values(
                                            by=['New Metric', 'Fraction matched disease HPO terms'],
                                            ascending=[False, False]
                                            )

            else:
                predicted_diseases = predict_disease_from_hpo(relevant_hpos[key], graph, ic_dict, semantic_similarity=semantic_similarity, weighted_score_active=weighted_score_active, neg_hpos=negative_hpos[key] if negative_hpos is not None else None)
                if orpha:
                    predicted_diseases_orpha = predict_disease_from_hpo(relevant_hpos[key], graph_orpha, ic_dict_orpha, OMIM_or_ORPHA='ORPHA', semantic_similarity=semantic_similarity, weighted_score_active=weighted_score_active, neg_hpos=negative_hpos[key] if negative_hpos is not None else None)
                    predicted_diseases = pd.concat([predicted_diseases, predicted_diseases_orpha])
                    predicted_diseases = predicted_diseases.sort_values(
                                            by=['New Metric', 'Fraction matched disease HPO terms'],
                                            ascending=[False, False]
                                            )

            no_of_diseases_considered.append(predicted_diseases.shape[0])
            predicted_diseases = predicted_diseases.head(20)
            
            # Log the file ID and correct diagnosis
            log_file.write(f"\nFile: {key}\n")
            try:
                log_file.write(f"Correct diagnosis: {prompts_df.loc[key, 'Correct Diagnosis'].strip()}\n")
            except Exception as e:
                log_file.write(f"Error fetching correct diagnosis: {e}\n")
                continue
            
            exact_match = False
            deepest_ancestor_match = False

            # Iterate through predictions
            for i, disease_id in enumerate(predicted_diseases.index):
                try:
                    disease_name = predicted_diseases.loc[disease_id]['Disease name']
                    log_file.write(f"{disease_name}\n")
                    # log_file.write(f"{predicted_diseases.loc[disease_id]}\n")
                    with pd.option_context(
                            "display.max_colwidth", None,          # no truncation of long strings
                            "display.max_seq_items", None,         # show full lists / sets
                            "display.width", None):                # don’t wrap into new lines
                        row_str = predicted_diseases.loc[disease_id].to_string()
                        log_file.write(f"{row_str}\n")

                    predictions_rows.append({
                        "file_id"           : key,
                        "rank"              : i+1,
                        "predicted_id"      : disease_id,
                        "predicted_name"    : disease_name,
                        "correct_diagnosis_id" : correct_diagnosis,
                        "correct_diagnosis_name" : prompts_df.loc[key, 'Correct Diagnosis'],
                        "case_description" : prompts_df.loc[key, 'Case Description'],
                        "patient_hpo_terms" : predicted_diseases.loc[disease_id]["Patient HPOs"],
                        "disease_hpo_terms" : predicted_diseases.loc[disease_id]["Disease HPOs"],
                        "overlap_hpo_terms" : predicted_diseases.loc[disease_id]["Overlap HPOs"],
                        "new_metric"        : predicted_diseases.loc[disease_id]["New Metric"],
                        "fraction_overlap"  : predicted_diseases.loc[disease_id]["Fraction overlapping search HPO terms"],
                        "fraction_matched"  : predicted_diseases.loc[disease_id]["Fraction matched disease HPO terms"],
                        "exact_match"       : False,            # will flip to True below
                        "deepest_ancestor"  : False             # will flip to True below

                    })

                    score = score_diagnosis(adapter, disease_id, gold_standard_id, depth_cache)
                    if score == "Exact match":
                        log_file.write(f"{score} Rank: {i+1}\n")
                        exact_match_ranks.append(i+1)
                        exact_match_ranks_dict[key] = i+1
                        exact_match = True
                        predictions_rows[-1]["exact_match"] = True
                        predictions_rows[-1]["deepest_ancestor"] = True
                        if not deepest_ancestor_match:
                            exact_match_deepest_ancestor_match_ranks.append(i+1)
                            exact_match_deepest_ancestor_match_ranks_dict[key] = i+1
                            deepest_ancestor_match = True
                            predictions_rows[-1]["deepest_ancestor"] = True
                        break
                    elif score == "Deepest ancestor match" and not deepest_ancestor_match:
                        log_file.write(f"{score} Rank: {i+1}\n")
                        deepest_ancestor_rank = i+1
                        exact_match_deepest_ancestor_match_ranks.append(deepest_ancestor_rank)
                        deepest_ancestor_match = True
                        predictions_rows[-1]["deepest_ancestor"] = True
                    else:
                        log_file.write(f"{score}\n")
                except Exception as e:
                    log_file.write(f"Error processing disease: {e}\n")
                    continue

            # Handle cases with no matches
            if not exact_match:
                exact_match_ranks.append(0)
                exact_match_ranks_dict[key] = 0
            if not deepest_ancestor_match:
                exact_match_deepest_ancestor_match_ranks.append(0)
                exact_match_deepest_ancestor_match_ranks_dict[key] = 0

    # Save the ranks to a file
    np.savez(output_ranks_path, 
             exact_match_ranks=exact_match_ranks, 
             exact_match_deepest_ancestor_match_ranks=exact_match_deepest_ancestor_match_ranks,
             no_of_diseases_considered=no_of_diseases_considered)
    
    # with open("case_description_exact_matches_gpt.json", "wb") as f:
    #     f.write(json.dumps(exact_match_ranks_dict).encode("utf-8"))

    # with open("case_description_deepest_ancestor_matches_gpt.json", "wb") as f:
    #     f.write(json.dumps(exact_match_deepest_ancestor_match_ranks_dict).encode("utf-8"))

    print(f"Logs saved to {log_file_path}")
    print(f"Ranks saved to {output_ranks_path}")

    pred_df = pd.DataFrame(predictions_rows)

    pred_df.to_csv(output_predictions_path, index=False)
    print(f"Prediction table saved → {output_predictions_path}")

def timed_evaluate(evaluate_function, *args, **kwargs):
    """
    Wrapper to measure and log execution time for any function.

    Args:
        evaluate_function (callable): The function to be timed.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The return value of the evaluate_function.
        float: The time taken in seconds.
    """
    start_time = time.time()
    result = evaluate_function(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    return result, elapsed_time