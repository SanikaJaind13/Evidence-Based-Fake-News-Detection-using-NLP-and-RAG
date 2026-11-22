
"""Named Entity Recognition using spaCy."""

import spacy

# Load the pretrained English NLP model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> dict:
    """Extract named entities from the given text."""

    if not text or not text.strip():
        return {"entities": [], "entity_counts": {}}

    doc = nlp(text)

    entities = []
    entity_counts = {}

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_)
        })

        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

    return {
        "entities": entities,
        "entity_counts": entity_counts
    }
