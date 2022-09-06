import string
import random
import time

# tokenizing the input sentence on the basis of the punctuations
# replace the punctuation marks by spaces
# and then split the sentence into words on the basis of the spaces
def tokenize(text): # returns List
    for punc in string.punctuation:
        text = text.replace(punc, ' '+punc+' ')
    t = text.split()
    return t

# this function returns the (n-1) grams as the "history sequence"
# remember! n grams means the next word is dependent on n-1 previous grams
# n-1 because the nth word is the word to be found itself - LOL
def get_ngrams(n, tokens):
    tokens = (n-1)*['<START>'] + tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return l
# above is the main logic behind ngrams

class N_GRAM(object):
    def __init__(self, n):
        self.n = n
        # a dict to keep list of candidate words given the context
        self.context = {}
        # keeps track of how many times an ngram has appeared in the text before
        self.ngram_counter = {}

    def update(self, sentence):
        # updates the language model
        n = self.n
        ngrams = get_ngrams(n, tokenize(sentence))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word) # append it to the existing entity
            else:
                self.context[prev_words] = [target_word] # create a new entity and put the target words later

    def probability(self, context, token):
        # calculates the probability of a candidate word for
        # a given context
        # this function does the count(w-2, w-1, w)/count(w-2, w-1) part
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        # this function is used to put a candidate word as the next word
        # for a given context sequence
        # we do this kind of in a semi random way
        # semi random because we choose the probability from the random function
        # comparing to that we put the candidate token if it has a probability
        # of more than the generated probability from the random

        r = random.random()
        maps_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            maps_to_probs[token] = self.probability(context, token)

        summ = 0

        for token in sorted(maps_to_probs):
            summ += maps_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count):
        # token_count = is the number of words you want to produce
        # return the generated text
        n = self.n
        context_queue = (n-1) * ['<START>']
        result = []

        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n-1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)

def ngram_model(n, path):
    m = N_GRAM(n)
    with open(path, 'r') as f:
        text = f.read()
        text = text.split('.')
        for sentence in text:
            sentence += '.'
            m.update(sentence)
    return m

if __name__ == "__main__":
    start = time.time()
    m = ngram_model(4, 'input_text.txt')

    print(f'Language Model creating time: {time.time() - start}')
    start = time.time()
    random.seed(7)
    print(f'{"="*50}\nGenerated text:')
    print(m.generate_text(20))
    print(f'{"="*50}')
