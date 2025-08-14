Unigram Model Perplexity: 3928.932498999517

Bigram Model Perplexity: undefined due to zero probability bigrams.

Smoothed Bigram Model Perplexity: 605.9443872907616

Question 1
Which model performed worst and why might you have
expected that model to have performed worst?

My bigram model performed the worst because it gave a lot of zero probabilities.
This happened because many word pairs in the test data did not appear in the
training data. Since the dataset is not huge and my text cleaning might have made
the vocabulary smaller or changed words, this made it harder for the bigram model
to find matches. It makes sense that the bigram model would do worse without
smoothing because it is more likely to see new pairs of words that it has never seen before.

Question 2
Did smoothing help or hurt the model’s ‘performance’
when evaluated on this corpus? Why might that be?

Smoothing really helped the model’s performance. Without smoothing, the bigram model
gave zero probabilities for unseen pairs which made it impossible to calculate perplexity.
Adding smoothing gave a small chance to every possible bigram, even the ones not seen in
training, so the model could calculate probabilities for all sentences. Also using log
sums helped avoid very small numbers that computers struggle to multiply. So smoothing
made the model more reliable on this test data.