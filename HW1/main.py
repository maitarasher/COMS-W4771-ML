import re
from collections import defaultdict
import random
import math

gpt_fn = 'humvgpt/gpt.txt'
hum_fn = 'humvgpt/hum.txt'

"""
Open files, store text in lower case
"""
gpt_f = open(gpt_fn, 'rt')
gpt_text = (gpt_f.read()).lower()
gpt_f.close()

hum_f = open(hum_fn, 'rt')
hum_text = (hum_f.read()).lower()
hum_f.close()

"""
remove all punctuation except “,.?!” 

source: https://www.educative.io/answers/remove-all-the-punctuation-marks-from-a-sentence-using-regex
"""
regex = r"[^\s\w\d,.?!]"
clean_gpt_text = re.sub(regex, '', gpt_text)
clean_hum_text = re.sub(regex, '', hum_text)

"""
split into data points
"""

gpt_texts = clean_gpt_text.split('\n')
hum_texts = clean_hum_text.split('\n')


random.shuffle(gpt_texts)
random.shuffle(hum_texts)

"""
Add <START> and <END> for each data point
"""
gpt_texts = ["<START> " + data + " <END>" for data in gpt_texts]
hum_texts = ["<START> " + data + " <END>" for data in hum_texts]


"""
Split to training data and testing data  
"""
index_gpt = int(len(gpt_texts) * 0.9)
index_hum = int(len(hum_texts) * 0.9)

gpt_training, gpt_testing = gpt_texts[:index_gpt], gpt_texts[index_gpt:]
hum_training, hum_testing = hum_texts[:index_hum], hum_texts[index_hum:]

"""
(ii) (b) 
Train a bigram and trigram model by finding the N-gram frequencies in the training
corpus. Calculate the percentage of bigrams/trigrams in the test set that do not appear in
the training corpus (this is called the OOV rate).

how to set default values for a dict: https://stackoverflow.com/questions/9139897/how-to-set-default-value-to-all-keys-of-a-dict-object-in-python
"""

# 0 index - gpt count, 1 index - hum count
# Construct unigrams dict as well to count frequencies so we can use it for bigram model
training_unigrams_counts = defaultdict(lambda: [0, 0])
training_bigrams_counts = defaultdict(lambda: [0, 0])
training_trigrams_counts = defaultdict(lambda: [0, 0])

# construct vocab for smoothing
vocab = set(clean_gpt_text.split() + clean_hum_text.split())

"""
Training - DICT
(*) gpt training - index 0
"""
for data in gpt_training:
    words = data.split()
    # traverse through words
    for i in range(len(words)):
        if (i == (len(words) - 1)):
            training_unigrams_counts[words[i]][0] += 1
        elif (i >= (len(words)) - 2):
            training_unigrams_counts[words[i]][0] += 1
            training_bigrams_counts[(words[i], words[i + 1])][0] += 1
        else:
            training_unigrams_counts[words[i]][0] += 1
            training_bigrams_counts[(words[i], words[i + 1])][0] += 1
            training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][0] += 1

"""
(**) hum training - index 1
"""
for data in hum_training:
    words = data.split()
    # traverse through words
    for i in range(len(words)):
        if (i == (len(words) - 1)):
            training_unigrams_counts[words[i]][1] += 1
        elif (i >= (len(words)) - 2):
            training_unigrams_counts[words[i]][1] += 1
            training_bigrams_counts[(words[i], words[i + 1])][1] += 1
        else:
            training_unigrams_counts[words[i]][1] += 1
            training_bigrams_counts[(words[i], words[i + 1])][1] += 1
            training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][1] += 1



"""
Testing - SET
(*) gpt testing
"""

test_bigrams = set()
test_trigrams = set()


for data in gpt_testing:
    words = data.split()
    # traverse through words
    for i in range((len(words)) - 1):
        if (i >= (len(words)) - 2):
            test_bigrams.add((words[i], words[i + 1]))
        else:
            test_bigrams.add((words[i], words[i + 1]))
            test_trigrams.add((words[i], words[i + 1], words[i + 2]))

"""
(**) hum testing 
"""

for data in hum_testing:
    words = data.split()
    # traverse through words
    for i in range((len(words)) - 1):
        if (i >= (len(words)) - 2):
            test_bigrams.add((words[i], words[i + 1]))
        else:
            test_bigrams.add((words[i], words[i + 1]))
            test_trigrams.add((words[i], words[i + 1], words[i + 2]))


#print(training_bigrams_counts[('cool','you')])
#print(training_trigrams_counts)

training_bigrams_keys = set(training_bigrams_counts.keys())
training_trigrams_keys = set(training_trigrams_counts.keys())

OOV_bigrams = len(test_bigrams.difference(training_bigrams_keys)) / len(training_bigrams_keys)
OOV_trigrams = len(test_trigrams.difference(training_trigrams_keys)) /  len(training_trigrams_keys)
#len((test_trigrams.union(training_trigrams_keys)))
print("OOV_bigrams:",OOV_bigrams)
print("OOV_trigrams:",OOV_trigrams)


""" 
(ii) (c) Evaluate the model on the test set and report the classification accuracy. Which model
performs better and why? Your justification should consider the bigram and trigram
OOV rate.
This study will also tell you how difficult or easy it is to distinguish human vs. AI
generated text!

P(Y|w1:n) 
"""

# y = 0 gpt, y = 1 hum
def bigram_classifier(text, y, py, training_bigrams_counts, training_unigrams_counts, V):
    log_count = [0,0]
    words = text.split()
    count_y = 0
    count_y_tag = 0
    for i in range((len(words)) - 1):
        # in order to avoid working with small probabilities, I will work with log.
        # log(xy) = log (x) + log(y)
        # this will not give me the probability but allow me to see what the bigger expression is
        """
        count_y *= ((training_bigrams_counts[(words[i], words[i + 1])][y] + 1) / (training_unigrams_counts[words[i]][y] + V))
        count_y *= ((training_bigrams_counts[(words[i], words[i + 1])][1 - y] + 1) / (training_unigrams_counts[words[i]][y] + V))
        """
        count_y += math.log((training_bigrams_counts[(words[i], words[i + 1])][y] + 1) / (training_unigrams_counts[words[i]][y] + V))
        count_y_tag += math.log((training_bigrams_counts[(words[i], words[i + 1])][1-y] + 1) / (training_unigrams_counts[words[i]][1 - y] + V))
    """
    prob = (py * count_y) / (py * count_y) + ((1-py) * count_y_tag)
    then if prob is greater than 0.5 I can return the class y, othervise 1-y
    if prob > 0.5:
        y_hat = y
    else:
        y_hat = 1-y
    """
    log_count[y] = math.log(py) + count_y
    log_count[1-y] = math.log(1 - py) + count_y_tag
    if (log_count[0] > log_count[1]):
        y_hat = 0
    else:
        y_hat = 1
    """prob = (math.log(py) + count_y) / ((py * count_y) + (math.log(1 - py) + count_y_tag))
    print(prob)"""
    return y_hat

def trigram_classifier(text, y, py, training_trigrams_counts, training_bigrams_counts, V):
    log_count = [0,0]
    words = text.split()
    count_y = 0
    count_y_tag = 0
    for i in range((len(words)) - 2):
        """
        count_y *= ((training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][y] + 1) / (training_bigrams_counts[(words[i], words[i + 1])][y] + V))
        count_y_tag *= ((training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][1 - y] + 1) / (training_bigrams_counts[(words[i], words[i + 1])][y] + V))
        """
        """
        In order to avoid working with small probabilities, we can work with log.
        # log(xy) = log (x) + log(y)
        # this will not give me the probability but allow me to see what the bigger expression is
        Using logs:
        """
        count_y += math.log((training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][y] + 1) / (training_bigrams_counts[(words[i], words[i + 1])][y] + V))
        count_y_tag += math.log((training_trigrams_counts[(words[i], words[i + 1], words[i + 2])][1-y] + 1) / (training_bigrams_counts[(words[i], words[i + 1])][1-y] + V))
    """
    prob = (py * count_y) / (py * count_y) + ((1-py) * count_y_tag)
    #then if prob is greater than 0.5 I can return the class y, otherwise 1-y
    if prob > 0.5:
        y_hat = y
    else:
        y_hat = 1-y
    """
    #Using logs:
    log_count[y] = math.log(py) + count_y
    log_count[1-y] = (math.log(1 - py) + count_y_tag)
    if (log_count[0] > log_count[1]):
        y_hat = 0
    else:
        y_hat = 1

    """prob = (math.log(py) + count_y) / ((py * count_y) + (math.log(1 - py) + count_y_tag))
    print(prob)"""
    return y_hat

# fraction of training data labelled as gpt
p0 = len(gpt_training) / (len(gpt_training) + len(hum_training))
p1 = 1 - p0
correct_predictions_bigram = 0
correct_predictions_trigram = 0

for data in gpt_testing:
    #P(Y = 0| data)
    y_hat_bigram = bigram_classifier(data, 0, p0, training_bigrams_counts, training_unigrams_counts, len(vocab))
    y_hat_trigram = trigram_classifier(data, 0, p0, training_trigrams_counts, training_bigrams_counts, len(vocab))
    if y_hat_bigram == 0:
        correct_predictions_bigram += 1
    if y_hat_trigram == 0:
        correct_predictions_trigram += 1

for data in hum_testing:
    # P(Y = 1| data)
    y_hat = bigram_classifier(data, 1, p1, training_bigrams_counts, training_unigrams_counts, len(vocab))
    y_hat_trigram = trigram_classifier(data, 0, p0, training_trigrams_counts, training_bigrams_counts, len(vocab))
    if y_hat_bigram == 1:
        correct_predictions_bigram += 1
    if y_hat_trigram == 0:
        correct_predictions_trigram += 1

acc_bigram_model = correct_predictions_bigram / (len(gpt_testing) + len(hum_testing))
acc_trigram_model = correct_predictions_trigram / (len(gpt_testing) + len(hum_testing))
print("acc_bigram_model:", acc_bigram_model)
print("acc_trigram_model:", acc_trigram_model)

"""
(iii)
Besides classification, N-gram models may also be used for text generation. 
Given a sequence of n − 1 previous tokens wi−n+2:i, the model selects the next word with probability
"""
def bigram_generation(T, num_words, model, text):
    """
    generated_text = []
    for i in range(num_words):
        words_seen = generated_text[-n:]
    """
    return 



