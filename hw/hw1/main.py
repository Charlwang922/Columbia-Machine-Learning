import string

# Define the filenames
hum_file = "hum.txt"
gpt_file = "gpt.txt"

# Define a function to clean the text
def clean_text(text):
    # Remove all punctuation except ,.?!
    translator = str.maketrans("", "", string.punctuation.replace(",.?!",""))
    text = text.translate(translator)
    # Convert to lower case
    text = text.lower()
    # Add special <START> and <END> tokens
    text = "<START> " + text + " <END>"
    return text

# Read in the human and GPT text files
with open(hum_file, encoding='utf-8') as f:
    hum_text = f.read()
with open(gpt_file, encoding='utf-8') as f:
    gpt_text = f.read()

# Clean the text
hum_text = clean_text(hum_text)
gpt_text = clean_text(gpt_text)

# Split the text into lists of words
hum_words = hum_text.split()
gpt_words = gpt_text.split()

# Partition 90% of the data to a training set and 10% to test set
hum_train_len = int(len(hum_words) * 0.9)
gpt_train_len = int(len(gpt_words) * 0.9)

hum_train = hum_words[:hum_train_len]
hum_test = hum_words[hum_train_len:]

gpt_train = gpt_words[:gpt_train_len]
gpt_test = gpt_words[gpt_train_len:]


print("Total number of words in the human train set:", len(hum_train))
print("Total number of words in the GPT train set:", len(gpt_train))
print("Total number of words in the human test set:", len(hum_test))
print("Total number of words in the GPT test set:", len(gpt_test))




from model import train_model

hum_bigram_freqs, hum_trigram_freqs, hum_oov_rate_bigrams, hum_oov_rate_trigrams = train_model(hum_train, hum_test)
gpt_bigram_freqs, gpt_trigram_freqs, gpt_oov_rate_bigrams, gpt_oov_rate_trigrams = train_model(gpt_train, gpt_test)
print(hum_oov_rate_bigrams, hum_oov_rate_trigrams, gpt_oov_rate_bigrams, gpt_oov_rate_trigrams)


from collections import Counter
hum_unigram_freqs = Counter(hum_train)
gpt_unigram_freqs = Counter(gpt_train)


import re
import math
from collections import Counter


# Build frequency tables for bigram and trigram models
hum_bigram_freqs = Counter(zip(hum_train[:-1], hum_train[1:]))
hum_trigram_freqs = Counter(zip(hum_train[:-2], hum_train[1:-1], hum_train[2:]))
gpt_bigram_freqs = Counter(zip(gpt_train[:-1], gpt_train[1:]))
gpt_trigram_freqs = Counter(zip(gpt_train[:-2], gpt_train[1:-1], gpt_train[2:]))

# Calculate conditional probabilities for bigram and trigram models
hum_bigram_probs = {}
hum_trigram_probs = {}
for bigram, count in hum_bigram_freqs.items():
    prev_word = bigram[0]
    hum_bigram_probs[bigram] = count / hum_unigram_freqs[prev_word]
for trigram, count in hum_trigram_freqs.items():
    prev_two_words = trigram[:2]
    hum_trigram_probs[trigram] = count / hum_bigram_freqs[prev_two_words]

gpt_bigram_probs = {}
gpt_trigram_probs = {}
for bigram, count in gpt_bigram_freqs.items():
    prev_word = bigram[0]
    gpt_bigram_probs[bigram] = count / gpt_unigram_freqs[prev_word]
for trigram, count in gpt_trigram_freqs.items():
    prev_two_words = trigram[:2]
    gpt_trigram_probs[trigram] = count / gpt_bigram_freqs[prev_two_words]




# Classify sentences in test sets as human or AI generated
test_probs = []
for i in range(len(hum_test)-1):
    bigram = tuple(hum_test[i:i+2])
    if bigram in hum_bigram_probs:
        p1 = hum_bigram_probs[bigram]
    else:
        p1 = 0
    if bigram in gpt_bigram_probs:
        p2 = gpt_bigram_probs[bigram]
    else:
        p2 = 0
    
    test_probs.append(p1 >= p2)

for i in range(len(gpt_test)-1):
    bigram = tuple(gpt_test[i:i+2])
    if bigram in hum_bigram_probs:
        p1 = hum_bigram_probs[bigram]
    else:
        p1 = 0
    if bigram in gpt_bigram_probs:
        p2 = gpt_bigram_probs[bigram]
    else:
        p2 = 0
    
    test_probs.append(p1 <= p2)

print(sum(test_probs) / len(test_probs))



# Classify sentences in test sets as human or AI generated
test_probs = []
for i in range(len(hum_test)-2):
    bigram = tuple(hum_test[i:i+3])
    if bigram in hum_trigram_probs:
        p1 = hum_trigram_probs[bigram]
    else:
        p1 = 0
    if bigram in gpt_trigram_probs:
        p2 = gpt_trigram_probs[bigram]
    else:
        p2 = 0
    
    test_probs.append(p1 >= p2)

for i in range(len(gpt_test)-2):
    bigram = tuple(gpt_test[i:i+3])
    if bigram in hum_trigram_probs:
        p1 = hum_trigram_probs[bigram]
    else:
        p1 = 0
    if bigram in gpt_trigram_probs:
        p2 = gpt_trigram_probs[bigram]
    else:
        p2 = 0
    
    test_probs.append(p1 <= p2)

print(sum(test_probs) / len(test_probs))