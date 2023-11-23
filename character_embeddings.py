import pandas as pd
from collections import defaultdict
from collections import Counter
from gensim.models import Word2Vec
from pypinyin import lazy_pinyin, Style

def generate_character_embeddings(datafile):
    '''
    Generate character embeddings using word2vec algorithm.
    Returns trained character embedding model.
    '''
    df = pd.read_csv(datafile)
    sentences = df.values.tolist()
    tokenized_sentences = []

    # tokenize data by character, one list of characters per sentence
    for sentence in sentences:
        tokenized_sentences.append(list("".join(sentence)))

    # generate word2vec embeddings
    model = Word2Vec(tokenized_sentences, vector_size=300)
    model.save('wv')

    return model

def group_homophones(char_embeddings):
    '''
    Group homophones (same pinyin and tone) from our trained embeddings.
    Returns dictionary with key:pinyin and value:list of embeddings that correspond to that pinyin.
    '''
    homophone_groups = defaultdict(list)
    chars = list(char_embeddings.wv.index_to_key)
    
    for char in chars:
        pinyin = lazy_pinyin(char, style=Style.TONE3)
        homophone_groups["".join(pinyin)].append((char, char_embeddings.wv[char]))

    # print groups of homophones
    for pinyin, embeddings in homophone_groups.items():
        if len(embeddings) > 1:
            print(f'{pinyin}: {[char for (char, _) in embeddings]}')

    return homophone_groups

def get_frequency(datafile):
    '''
    Returns dictionary with key: frequency and value: list of characters that have that frequency
    '''
    df = pd.read_csv(datafile)
    frequencies = Counter()
    reverse_frequencies = defaultdict(list)
    sentences = df.values.tolist()
    tokenized_sentences = []

    # tokenize data by character, one list of characters per sentence
    for sentence in sentences:
        tokenized_sentences.append(list("".join(sentence)))

    for sentence in tokenized_sentences:
        frequencies.update(sentence)

    # map frequency dict to reverse frequency dict
    for word, freq in frequencies.items():
        reverse_frequencies[freq].append(word)
    
    return reverse_frequencies

def compute_distance(h1, h2, all_embeddings, frequency_dict):
    pass

if __name__ == "__main__":
    # print(group_homophones(generate_character_embeddings('transcripts.tsv')))
    print(get_frequency('transcripts.tsv'))