
from src.ner import extract_entities

text = """
The Indian government announced a renewable energy project
in Maharashtra on 15 November 2026.
Prime Minister Narendra Modi attended the event in Mumbai.
"""

result = extract_entities(text)

print("Extracted Entities:")
for entity in result["entities"]:
    print(entity["text"], "-", entity["label"],
          "-", entity["description"])

print("\nEntity Counts:")
print(result["entity_counts"])
