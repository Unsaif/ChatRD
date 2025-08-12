import numpy as np
from pyhpo import Ontology  # Ensure pyhpo is set up and initialized
import pandas as pd

# Initialize pyhpo Ontology
_ = Ontology()

def get_disease_profile(hpo_data, disease_name=None, disease_id=None):
    """
    Reads the HPO phenotype.hpoa file and constructs a disease profile
    with HPO terms and their frequencies for a specific disease using Pandas.
    
    Parameters:
    - filepath: Path to the phenotype.hpoa file
    - disease_name: Name of the disease to look up (optional)
    - disease_id: ID of the disease to look up (optional)

    Returns:
    - profile: Dictionary where keys are HPO term IDs and values are frequencies
    """

    # Filter rows based on disease name or disease ID
    if disease_id:
        disease_data = hpo_data[hpo_data['database_id'] == disease_id]
    elif disease_name:
        disease_data = hpo_data[hpo_data['disease_name'].str.lower() == disease_name.lower()]
    else:
        raise ValueError("Please specify either disease_name or disease_id.")

    # Extract HPO ID and Frequency, filling missing frequencies with 'Unknown'
    profile = disease_data[['hpo_id', 'frequency']].fillna('Unknown')
    
    # Convert to dictionary format: {HPO_ID: Frequency}
    profile_dict = dict(zip(profile['hpo_id'], profile['frequency']))
    
    return profile_dict

# Define the helper functions
def get_phenotype_name(hpo_id):
    term = Ontology.get_hpo_object(hpo_id)
    return term.name if term else "Unknown Phenotype"

def convert_frequency(value, frequency_map):
    if value in frequency_map:
        return frequency_map[value]
    elif "/" in value:
        try:
            numerator, denominator = map(int, value.split("/"))
            return round((numerator / denominator) * 100)
        except (ValueError, ZeroDivisionError):
            return np.nan
    elif value != "Unknown":
        try:
            return get_phenotype_name(value)
        except:
            return "Unknown"
    else:
        return "Unknown"
    
def aggregate_embeddings_average(hpos, collection):
    # Ensure the input is a non-empty list
    if not hpos or not isinstance(hpos, list) or not all(isinstance(i, str) for i in hpos):
        return None

    # Retrieve embeddings in one batch call
    results = collection.get(where={"Id": {"$in": hpos}}, include=["embeddings"])
    
    # Check if embeddings are present
    embeddings = results.get('embeddings', [])
    if embeddings is None or len(embeddings) == 0:
        return None

    # Calculate the average embedding
    embeddings_array = np.array(embeddings)
    aggregated_embedding = np.mean(embeddings_array, axis=0)
    return aggregated_embedding

def retrieve_nearest_neighbours(
        hpo_terms,
        collection, 
        profiles,       # ← Chroma collection that holds your profiles
        id_to_disease=None,# optional: map Chroma IDs → disease names
        k=1000
):
    """
    Given a list of HPO terms, return up to k nearest‑neighbour diseases
    from the Chroma collection.
    
    Parameters
    ----------
    hpo_terms : list[str]
        HPO codes describing the patient.
    collection : chromadb.api.models.Collection
        The Chroma collection that contains the disease profile vectors.
    id_to_disease : dict[str, str] or None
        Optional mapping from Chroma IDs to disease names.  Provide this
        if the collection’s `documents` field is *not* the disease name.
    k : int, default 1000
        Number of neighbours to retrieve.
    
    Returns
    -------
    list[str]
        Unique disease names ordered by increasing vector distance.
    """

    # 1) Build the query vector exactly like the inserts
    query_vec = aggregate_embeddings_average(hpo_terms, collection)
    if isinstance(query_vec, np.ndarray):
        query_vec = query_vec.astype(float).tolist()   # Chroma expects list

    # 2) HNSW search inside Chroma
    res = profiles.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "distances"]
    )

    # 3) Turn IDs → disease names
    # if id_to_disease is not None:
    #     diseases  = [id_to_disease[i] for i in res["ids"][0]]
    # else:
    #     diseases  = res["documents"][0]   # you stored the name as document
    diseases  = res["documents"][0]
    distances = res["distances"][0]

    # 4) Put in a DataFrame just like before
    results_df = pd.DataFrame({
        "Disease":  diseases,
        "Distance": distances
    })

    return results_df["Disease"].unique().tolist()

# def retrieve_nearest_neighbours(hpo_terms, index, id_to_disease, collection, k=1000):

#     new_embedding = aggregate_embeddings_average(hpo_terms, collection)

#     # Ensure the new embedding is a NumPy array of type float32
#     new_embedding = np.array(new_embedding).astype('float32')

#     # Query the index for the 100 nearest neighbors
#     labels, distances = index.knn_query(new_embedding, k=k)

#     # Retrieve the corresponding disease names
#     nearest_diseases = [id_to_disease[id] for id in labels[0]]

#     # Optionally, get distances
#     nearest_distances = distances[0]

#     # Create a DataFrame of results
#     results_df = pd.DataFrame({
#         'Disease': nearest_diseases,
#         'Distance': nearest_distances
#     })

#     return list(results_df['Disease'].unique()) 

