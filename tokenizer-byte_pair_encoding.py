# reference: https://huggingface.co/learn/nlp-course/en/chapter6/5
from transformers import AutoTokenizer
from collections import defaultdict

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    for word, offset in words_with_offset:
        # counting the word freqency. We do not need the offset
        word_freqs[word] += 1

print(word_freqs)


# counting the unique letters
alphabet = set()
for word in word_freqs.keys():
    for char in word:
        alphabet.add(char)

alphabet = list(alphabet)
alphabet.sort()
print(alphabet)

vocabulary = ["<|endoftext|>"] + alphabet
print(vocabulary)

# split the words into chars
splits = {w: [c for c in w] for w in word_freqs.keys()}
print(splits)

# compute the pair frequencies
def compute_pair_freqs(splits):
    pair_freqencies = defaultdict(int)

    for word in word_freqs.keys():
        split = splits[word]

        # the following case does not contain a pair of alphabets
        if len(split)<=1:
            continue

        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freqencies[pair] += 1

    return pair_freqencies

pair_freqs = compute_pair_freqs(splits)

# checking the first 5 pairs
for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        print("-"*70)
        break

# Testing the most frequent pairs
best_pair = None
max_freq = 0

for pair, freq in pair_freqs.items():
    if freq>max_freq:
        best_pair = pair
        max_freq = freq
print(best_pair, max_freq)


def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]

        if len(split)<=1:
            continue
        i = 0
        while i<len(split)-1:
            if split[i]==a and split[i+1]==b:
                # excluding the previous 2 by merging and adding the pair as one
                split = split[:i]+[a+b]+split[i+2:]
            else:
                i+=1
        splits[word] = split
    return splits

# testing merge pair
splits = merge_pair("Ġ", "t", splits)
print(splits["Ġtrained"])


vocab_size = 70

merge_rules = {}

while(len(vocabulary) < vocab_size):
    pair_freqs = compute_pair_freqs(splits)

    # computing best(max) pair freq
    best_pair = None
    max_freq = 0
    for pair, freq in pair_freqs.items():
        if freq >= max_freq:
            best_pair = pair
            max_freq = freq

    # merging best pairs in each iteration
    splits = merge_pair(*best_pair, splits)

    # storing the merge rule-> required for tokenization
    merge_rules[best_pair] = best_pair[0]+best_pair[1]
    vocabulary.append(best_pair[0]+best_pair[1])

print("Vocabs: ", vocabulary)
print("Merges: ", merge_rules)


def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_words = [word for word, offset in pre_tokenize_result]
    splits = [[char for char in word] for word in pre_tokenized_words]

    for pair, merge in merge_rules.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split)-1:
                if split[i]==pair[0] and split[i+1]==pair[1]:
                    split = split[:i]+[merge]+split[i+2:]
                else:
                    i+=1
            splits[idx] = split
    return sum(splits,[])

print(tokenize("This is not a token."))

