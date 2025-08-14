from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import random


class NgramPreprocessor:
    def __init__(self, filename, train_ratio=0.8):
        with open(filename, 'r') as f:
            text = f.read().lower()

        # set seed for consistent random shuffling
        random.seed(456)

        # tokenize sentences and shuffle
        sentences = sent_tokenize(text)
        random.shuffle(sentences)

        # split train/test
        split = int(len(sentences) * train_ratio)
        self.training_sentences = sentences[:split]
        self.testing_sentences = sentences[split:]

        # use RegexpTokenizer to remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')

        # tokenize into words
        self.training_words = [['<s>'] + tokenizer.tokenize(s) + ['</s>'] for s in self.training_sentences]
        self.testing_words = [['<s>'] + tokenizer.tokenize(s) + ['</s>'] for s in self.testing_sentences]

        # flattened lists
        self.flat_training_words = [word for sent in self.training_words for word in sent]
        self.flat_testing_words = [word for sent in self.testing_words for word in sent]

    # returns a list of all bigrams
    def get_bigrams(self):
        return list(zip(self.flat_training_words[:-1], self.flat_training_words[1:]))

    # returns a list of all trigrams
    def get_trigrams(self):
        return list(zip(self.flat_training_words[:-2],
                        self.flat_training_words[1:-1],
                        self.flat_training_words[2:]))