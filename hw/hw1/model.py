import collections

def train_model(train, test):
    # Calculate bigram and trigram frequencies in training set
    bigram_freqs = collections.defaultdict(int)
    trigram_freqs = collections.defaultdict(int)
    for i in range(len(train)-1):
        bigram = tuple(train[i:i+2])
        bigram_freqs[bigram] += 1
    for i in range(len(train)-2):
        trigram = tuple(train[i:i+3])
        trigram_freqs[trigram] += 1

    # Calculate total number of bigrams and trigrams in training set
    num_bigrams = sum(bigram_freqs.values())
    num_trigrams = sum(trigram_freqs.values())

    # Calculate OOV rate for bigrams and trigrams in test set
    num_oov_bigrams = 0
    num_oov_trigrams = 0
    for i in range(len(test)-1):
        bigram = tuple(test[i:i+2])
        if bigram not in bigram_freqs:
            num_oov_bigrams += 1
    for i in range(len(test)-2):
        trigram = tuple(test[i:i+3])
        if trigram not in trigram_freqs:
            num_oov_trigrams += 1

    oov_rate_bigrams = num_oov_bigrams / len(test)
    oov_rate_trigrams = num_oov_trigrams / len(test)

    return bigram_freqs, trigram_freqs, oov_rate_bigrams, oov_rate_trigrams

