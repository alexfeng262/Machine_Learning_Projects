import collections
import nltk
import os
import random

# NOT WORKING AT ALL

# nltk.download('wordnet')


def load_files(directory):
    result = []
    for fname in os.listdir(directory):
        with open(directory + fname, 'r', errors='ignore') as f:
            result.append(f.read())
    return result


def preprocess_sentence(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    # clearly list out our preprocessing pipeline
    processed_tokens = nltk.word_tokenize(sentence)
    processed_tokens = [w.lower() for w in processed_tokens]
    # find least common elements
    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]
    # remove these tokens
    # processed_tokens = [w for w in processed_tokens if w not in stop_words]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    # lemmatize
    processed_tokens = [lemmatizer.lemmatize(w) for w in processed_tokens]
    return processed_tokens


def feature_extraction(tokens):
    return dict(collections.Counter(tokens))


def train_test_split(dataset, train_size=0.8):
    num_training_examples = int(len(dataset) * train_size)
    return dataset[:num_training_examples], dataset[num_training_examples:]


positive_examples = load_files('enron/spam/')
negative_examples = load_files('enron/ham/')

# Label the examples
positive_examples = [preprocess_sentence(email) for email in positive_examples]
positive_examples = [preprocess_sentence(email) for email in negative_examples]

positive_examples = [(email, 1) for email in positive_examples]
negative_examples = [(email, 0) for email in negative_examples]

all_examples = positive_examples + negative_examples
random.shuffle(all_examples)
print(all_examples[0])
featurized = [(feature_extraction(corpus), label) for corpus, label in all_examples]

print(featurized[0])

training_set, test_set = train_test_split(featurized)

model = nltk.classify.NaiveBayesClassifier.train(training_set)

training_error = nltk.classify.accuracy(model, training_set)
print('Model training complete. Acurracy on training set: {}'.format(training_error))

testing_error = nltk.classify.accuracy(model, test_set)
print('Model training complete. Acurracy on test set: {}'.format(testing_error))



