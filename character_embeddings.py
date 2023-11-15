import pandas as pd
from gensim.models import Word2Vec

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
        tokenized_sentences.append(list(sentence)) # ok why is it treating the sentence as one unit

    # apply word2vec to data
    model = Word2Vec(tokenized_sentences, size=300)
    model.save('wv')

def compute_distance(h1, h2, all):
    pass

if __name__ == "__main__":
    generate_character_embeddings('transcripts.tsv')