import pandas as pd
from collections import defaultdict
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from pypinyin import lazy_pinyin, Style

class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 20:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step)) # only print final loss after completing all epochs
            # print('Perplexity after epoch {}: {}'.format(self.epoch, 2 ** (loss - self.loss_previous_step)))
            # gonna keep this commented out until we get low enough loss to justify printing perplexity....
        self.epoch += 1
        self.loss_previous_step = loss

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
    model = Word2Vec(tokenized_sentences, vector_size=300, epochs=20, compute_loss=True, callbacks=[callback()])
    model.save('anita/small-wv')

    return model

def generate_pinyin_embeddings(datafile):
    '''
    Generate pinyin embeddings using word2vec algorithm.
    Returns trained pinyin embedding model.
    '''
    df = pd.read_csv(datafile)
    sentences = df.values.tolist()
    tokenized_sentences = []

    for sentence in sentences:
        characters = list("".join(sentence))
        pinyins = []

        for char in characters:
            pinyin = lazy_pinyin(char, style=Style.TONE3)
            pinyins.extend(pinyin)

        tokenized_sentences.append("".join(pinyins))

    model = Word2Vec(tokenized_sentences, vector_size=300, epochs=20, compute_loss=True, callbacks=[callback()])
    model.save('anita/wv-pinyin')

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
        homophone_groups["".join(pinyin)].append(char)

    # print groups of homophones
    # for pinyin, embeddings in homophone_groups.items():
    #     if len(embeddings) > 1:
    #         print(f'{pinyin}: {[char for (char, _) in embeddings]}')

    return homophone_groups

def get_frequency(datafile):
    '''
    Returns (1) dictionary with key: character and value: frequency of that character
            (2) dictionary with key: frequency and value: list of characters that have that frequency.
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
    
    return frequencies, reverse_frequencies

def compute_distance(homophone_groups, all_embeddings, frequency_dict, reverse_frequency_dict):
    '''
    Prints average baseline similary, average homohpone group similarity, and difference between the two.
    '''
    baseline_similarity = 0
    baseline_counter = 0
    homophone_similarity = 0
    homophone_counter = 0

    for homophones in homophone_groups.values():
        homophones = set(homophones)
        if len(homophones) > 1:
            for h1 in homophones: # compute average homophone pair distance and baseline distance for each homophone
                for h2 in homophones:
                    if h1 != h2:
                        # compute baseline similarities
                        frequency = frequency_dict[h2]
                        similar_frequency_chars = reverse_frequency_dict[frequency]

                        if len(similar_frequency_chars) == 0: # no characters of same frequency, so check nearby frequencies
                            for i in range(1, 100):
                                if len(reverse_frequency_dict[frequency + i]) != 0:
                                    similar_frequency_chars = reverse_frequency_dict[frequency + i]
                                    break
                                elif len(reverse_frequency_dict[frequency - i]) != 0:
                                    similar_frequency_chars = reverse_frequency_dict[frequency - i]
                                    break

                        for similar_frequency_char in similar_frequency_chars:
                            if similar_frequency_char not in homophones:
                                baseline_similarity += all_embeddings.wv.similarity(h1, similar_frequency_char)
                                baseline_counter += 1

                        # compute homophone similarities
                        homophone_similarity += all_embeddings.wv.similarity(h1, h2)
                        homophone_counter += 1

    baseline_average = baseline_similarity / baseline_counter
    homophone_average = homophone_similarity / homophone_counter
    print(f'Average Baseline Similarity: {baseline_average}')
    print(f'Average Homophone Group Similarity: {homophone_average}')
    print(f'Difference Between Average Baseline and Homophone Similarities: {baseline_average - homophone_average}')

if __name__ == "__main__":
    embeddings = generate_character_embeddings('data/transcripts-12k.tsv')
    # homophone_groups = group_homophones(embeddings)
    # frequency_dict, reverse_frequency_dict = get_frequency('data/transcripts-12k.tsv')

    # compute_distance(homophone_groups, embeddings, frequency_dict, reverse_frequency_dict)
    # generate_pinyin_embeddings('data/usable-dev.txt')