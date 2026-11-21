
from src.config import load_settings
from src.summarizer import summarize_article


sample_article = """
The Indian government announced a new initiative to promote
artificial intelligence education in schools and colleges.
The initiative aims to provide AI learning resources,
teacher training and practical learning opportunities.
The programme will initially focus on selected institutions
before expanding to more educational institutions across India.
"""

settings = load_settings()

result = summarize_article(sample_article, settings)

print("\n========== ARTICLE SUMMARY ==========\n")
print(result)
