import math
from preprocessor import NgramPreprocessor
from collections import Counter

processor = NgramPreprocessor("doyle_Bohemia.txt")

# create a unigram vocab Counter()
unigram_vocab = Counter(processor.flat_training_words)
bigram_vocab = Counter(processor.get_bigrams())

bigram_probabilities = {
    bigram: count / unigram_vocab[bigram[0]]
    for bigram, count in bigram_vocab.items()
}

with open('bigram_probs.txt', 'w') as f:
    for bigram, prob in bigram_probabilities.items():
        f.write(f"p({bigram[1]} | {bigram[0]}) = {prob}\n")

# evaluate sentence probabilities on test data
sentence_probabilities = {}

for words in processor.testing_words:
    sentence = ' '.join(words)
    sentence_probability = 1.0

    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        prob = bigram_probabilities.get(bigram, 0.0)
        sentence_probability *= prob
    sentence_probabilities[sentence] = sentence_probability

with open('bigram_eval.txt', 'w') as f:
    for sentence, prob in sentence_probabilities.items():
        f.write(f"p({sentence}) = {prob}\n")


# math for perplexity
log_sum = 0.0
total_tokens = 0

for words in processor.testing_words:
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        prob = bigram_probabilities.get(bigram, 0.0)
        if prob > 0:
            log_sum += math.log(prob)
        else:
            log_sum += float('-inf')  # Optional: logs -inf if unseen bigram
        total_tokens += 1

print(f"Log-sum: {log_sum}")

if total_tokens > 0 and log_sum != float('-inf'):
    perplexity = math.exp(-log_sum / total_tokens)
    print(f"Perplexity: {perplexity}")
else:
    print("Perplexity: undefined due to zero probability bigrams.")
