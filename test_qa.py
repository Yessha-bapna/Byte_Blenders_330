# test_qa.py

from app.services.qa_engine import generate_answer

question = "Does this policy cover air ambulance services?"
response = generate_answer(question)
print("\nGPT Response:\n", response)
