
import spacy

nlp = spacy.load("en_core_web_sm")

def parse_food_name(food_name):
    """Extract meaningful components from food name using NLP."""
    doc = nlp(food_name.lower())
    return [token.lemma_ for token in doc if not token.is_stop]
    