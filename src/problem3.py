import math
from preprocessor import NgramPreprocessor
from collections import Counter

# smoothing parameter
delta = 0.1

processor = NgramPreprocessor("doyle_Bohemia.txt")

# create a unigram vocab Counter()
unigram_vocab = Counter(processor.flat_training_words)

# create a bigram vocab Counter()
bigram_vocab = Counter(processor.get_bigrams())

vocab_size = len(unigram_vocab)

# create bigram probs with Add-k smoothing (add a delta of 0.1)
bigram_probabilities = {
    bigram: (count + delta) / (unigram_vocab[bigram[0]] + delta * vocab_size)
    for bigram, count in bigram_vocab.items()
}

with open('smooth_probs.txt', 'w') as f:
    for bigram, prob in bigram_probabilities.items():
        f.write(f"p({bigram[1]} | {bigram[0]}) = {prob}\n")

# evaluate sentence probabilities on test data
sentence_probabilities = {}

for words in processor.testing_words:
    sentence = ' '.join(words)
    sentence_probability = 1.0

    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        count = bigram_vocab.get(bigram, 0)
        prob = (count + delta) / (unigram_vocab[bigram[0]] + delta * vocab_size)
        sentence_probability *= prob
    sentence_probabilities[sentence] = sentence_probability

with open('smoothed_eval.txt', 'w') as f:
    for sentence, prob in sentence_probabilities.items():
        f.write(f"p({sentence}) = {prob}\n")


# math for perplexity
log_sum = 0.0
total_tokens = 0

for words in processor.testing_words:
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        count = bigram_vocab.get(bigram, 0)
        prob = (count + delta) / (unigram_vocab[bigram[0]] + delta * vocab_size)
        log_sum += math.log(prob)
        total_tokens += 1

print(f"Log-sum: {log_sum}")

if total_tokens > 0:
    perplexity = math.exp(-log_sum / total_tokens)
    print(f"Perplexity: {perplexity}")
else:
    print("Perplexity: undefined (no tokens)")
