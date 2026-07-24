from transformers import BertTokenizer, RobertaTokenizer
import pandas as pd

# Load tokenizers
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Self-created sample sentences
sentences = [
    "I love learning Natural Language Processing.",
    "Transformers have changed modern AI research.",
    "ChatGPT can generate human-like responses.",
    "Computer vision models detect objects in images.",
    "The unbelievable performance amazed everyone.",
    "Tokenization splits text into smaller pieces.",
    "Deep learning requires large amounts of data.",
    "Hyderabad is becoming an AI innovation hub.",
    "My neural network achieved 98% accuracy.",
    "Self-driving cars rely on sensors and cameras."
]

results = []

for i, sentence in enumerate(sentences, start=1):
    bert_tokens = bert_tokenizer.tokenize(sentence)
    roberta_tokens = roberta_tokenizer.tokenize(sentence)

    results.append({
        "Sentence ID": f"S{i}",
        "Sentence": sentence,
        "BERT Tokens": bert_tokens,
        "BERT Count": len(bert_tokens),
        "RoBERTa Tokens": roberta_tokens,
        "RoBERTa Count": len(roberta_tokens)
    })

# Create DataFrame
df = pd.DataFrame(results)

# Display results
print(df.to_string(index=False))    