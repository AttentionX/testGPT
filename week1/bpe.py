import re
import collections
from pprint import pprint
import matplotlib.pyplot as plt

num_merges = 100

dictionary = {
    'a t t e n t i o n </w>': 1,
    't o k e n i z a t i o n </w>': 1,
    'b a t c h </w>': 1,
    't r a n s f o r m e r s </w>': 1,
    'b i g r a m </w>': 1,
    'l a n g u a g e </w>': 1,
    'm o d e l </w>': 1,
    'l o s s </w>': 1,
    't r a i n </w>': 1,
    'm a t r i x </w>': 1,
    'm u l t i p l y </w>': 1,
    's o f t m a x </w>': 1,
    'p o s i t i o n a l </w>': 1,
    'e n c o d e r </w>': 1,
    'd e c o d e r </w>': 1,
    's q r t </w>': 1,
    'b y t e </w>': 1,
    'p a i r </w>': 1,
    'e n c o d i n g </w>': 1,
    'c o n t e x t </w>': 1,
    'a v e r a g e </w>': 1
}


def get_stats(dictionary):
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_dictionary(pair: dict[tuple[str, str], int], v_in: dict[str, int]):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_vocabulary(dictionary: dict[str, int]):
    vocabulary = set()
    for word, freq in dictionary.items():
        symbols = word.split()
        vocabulary = vocabulary.union(set(symbols))
    return vocabulary


bpe_codes = {}
bpe_codes_reverse = {}
dict_length_history = [len(get_vocabulary(dictionary))]

for i in range(num_merges):
    pairs = get_stats(dictionary)
    best = max(pairs, key=pairs.get)
    dictionary = merge_dictionary(best, dictionary)

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best
    vocabulary = get_vocabulary(dictionary)
    dict_length_history.append(len(vocabulary))
    if i % 10 == 0:
        pprint(vocabulary)

plt.plot(dict_length_history)
plt.xlabel('num of iteration')
plt.ylabel('vocabulary')
plt.title('Byte Pair Encoding')
plt.show()
