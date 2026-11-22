from src.preprocessing import preprocess_text

sample = """
India's GDP grew by 7.5% in 2023.
The project cost ₹10,000 and covered 5.5 km.
"""

result = preprocess_text(sample)

print("\nCLEANED TEXT:", result["cleaned_text"])
print("\nTOKENS:", result["tokens"])
print("\nPROCESSED TEXT:", result["processed_text"])