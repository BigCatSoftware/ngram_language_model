import math
from preprocessor import NgramPreprocessor
from collections import Counter

processor = NgramPreprocessor("doyle_Bohemia.txt")

# create vocab Counter(), a subclass of dictionary
unigram_vocab = Counter(processor.flat_training_words)

# get a total word count for unigram model probability
total_word_count = sum(unigram_vocab.values())

# create unigram prob model
probabilities = {word: count / total_word_count for word, count in unigram_vocab.items()}

# print unigram probabilities to unigram_probs.txt
with open('unigram_probs.txt', 'w') as f:
    for word, prob in probabilities.items():
        f.write(f"p({word}) = {prob}\n")


sentence_probabilities = {}

for words in processor.testing_words:
    sentence = ' '.join(words)
    sentence_probability = 1.0
    for word in words:
        word_prob = probabilities.get(word, 0.0) #probability is 0 if word not seen
        sentence_probability *= word_prob
    sentence_probabilities[sentence] = sentence_probability

with open('unigram_eval.txt', 'w') as f:
    for sentence, prob in sentence_probabilities.items():
        f.write(f"p({sentence}) = {prob}\n")

# math for perplexity function with the given sentence probabilities
# total number of words in the test set
total_words = sum(len(words) for words in processor.testing_words)

log_sum = 0.0
for words in processor.testing_words:
    for word in words:
        word_prob = probabilities.get(word, 1e-12)  # use small floor prob for unknown words
        log_sum += math.log(word_prob)

avg_log_prob = log_sum / total_words
perplexity = math.exp(-avg_log_prob)

print(f"Log-sum: {log_sum}")
print(f"Perplexity: {perplexity}")