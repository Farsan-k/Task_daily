import random
from collections import defaultdict

# ==========================
# STEP 1 : Read Training Data
# ==========================

with open("text.txt", "r") as file:
    sentences = file.read().lower().splitlines()

print("Training Sentences:")
print(sentences)

# ==========================
# STEP 2 : Tokenization
# ==========================

tokenized_sentences = []

for sentence in sentences:
    words = sentence.split()
    tokenized_sentences.append(words)

print("\nTokenized Sentences")
print(tokenized_sentences)

# ==========================
# STEP 3 : Build Vocabulary
# ==========================

vocabulary = set()

for sentence in tokenized_sentences:
    for word in sentence:
        vocabulary.add(word)

print("\nVocabulary")
print(sorted(vocabulary))

# ==========================
# STEP 4 : Learn Next Word
# ==========================

model = defaultdict(list)

for sentence in tokenized_sentences:

    for i in range(len(sentence)-1):

        current_word = sentence[i]
        next_word = sentence[i+1]

        model[current_word].append(next_word)

print("\nLearned Model")

for key in model:
    print(key, "->", model[key])

# ==========================
# STEP 5 : Generate Text
# ==========================

def generate_text(start_word, length=10):

    current_word = start_word.lower()

    sentence = [current_word]

    for _ in range(length):

        if current_word not in model:
            break

        next_word = random.choice(model[current_word])

        sentence.append(next_word)

        current_word = next_word

    return " ".join(sentence)

# ==========================
# STEP 6 : User Input
# ==========================

while True:

    prompt = input("\nEnter first word (or quit): ")

    if prompt.lower() == "quit":
        break

    output = generate_text(prompt)

    print("\nGenerated Text:")
    print(output)