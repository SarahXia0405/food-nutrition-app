import spacy
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load medium-sized spaCy model (better than en_core_web_sm)
nlp = spacy.load("en_core_web_md")

# Load food components from pickle
with open("food_components.pkl", "rb") as f:
    all_components = pickle.load(f)

# Load nutrition data
component_nutrition_contributions = pd.read_csv("component_nutrition_contributions.csv", index_col=0)

# Precompute word vectors for all known food components
component_vectors = {}
for word in all_components:
    word_vector = nlp(word).vector
    if np.any(word_vector):  # Ensure word has a vector (avoid OOV issues)
        component_vectors[word] = word_vector

# Save precomputed vectors for fast reuse
with open("component_vectors.pkl", "wb") as f:
    pickle.dump(component_vectors, f)

def parse_food_name(food_name):
    '''Extract meaningful words from food names.'''
    doc = nlp(food_name.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def find_closest_component(user_input):
    '''Find the closest matching food component using NLP embeddings.'''
    user_words = parse_food_name(user_input)  
    matched_words = []

    for word in user_words:
        # Step 1: Check for an exact match
        if word in all_components:
            matched_words.append(word)
            continue  # Skip similarity check if exact match found

        # Step 2: Compute similarity with known components
        word_vector = nlp(word).vector
        if not np.any(word_vector):  # Skip words without a vector
            continue

        word_vector = word_vector.reshape(1, -1)

        similarities = {
            comp: cosine_similarity(word_vector, comp_vec.reshape(1, -1))[0][0]
            for comp, comp_vec in component_vectors.items()
        }

        # Get best match
        best_match = max(similarities, key=similarities.get)
        best_match_score = similarities[best_match]

        # Avoid poor matches (threshold > 0.5)
        if best_match_score > 0.5:
            matched_words.append(best_match)

    return matched_words
