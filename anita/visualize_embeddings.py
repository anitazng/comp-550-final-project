from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np
import matplotlib
import hanlp
from pypinyin import lazy_pinyin, Style
from collections import defaultdict

word2vec = hanlp.load(hanlp.pretrained.word2vec.RADICAL_CHAR_EMBEDDING_100)

def group_true_homophones_from_file(charfile):
    '''
    Reads in file with characters.
    Group true homophones (same pinyin and tone) from characters.
    Returns dictionary with key:pinyin and value:list of characters that correspond to that pinyin.
    '''
    homophone_groups = defaultdict(list)
    with open(charfile, 'r') as file:
            lines = file.readlines()
            chars = [line[0] for line in lines]
    
    for char in chars:
        pinyin = lazy_pinyin(char, style=Style.TONE3) # take tone into account
        homophone_groups["".join(pinyin)].append(char)

    # print groups of homophones
    # for pinyin, embeddings in homophone_groups.items():
    #     if len(embeddings) > 1:
    #         print(f'{pinyin}: {[char for (char, _) in embeddings]}')

    return homophone_groups

def get_homophone_group_embeddings():
    '''
    Returns list of homophone group dictionaries (key: character, value: embedding)
    '''
    homophone_groups = group_true_homophones_from_file('data/pretrained/words.txt')
    group_embeddings = []
    
    for chars in homophone_groups.values():
        group = {}
        if len(chars) > 1:
            for char in chars:
                embedding_tensor = word2vec(char)
                # Convert PyTorch tensor to NumPy array
                embedding_array = embedding_tensor.detach().numpy()
                group[char] = embedding_array
        
            group_embeddings.append(group)
    
    return group_embeddings

def get_frequency_list(charfile):
    '''
    Returns list of characters in order of decreasing frequency according to wordfreq.
    '''
    with open(charfile, 'r') as file:
            lines = file.readlines()
            chars = [line[0] for line in lines]

    return chars

if __name__ == "__main__":
    # Extract embeddings and characters lists
    freq_list = get_frequency_list('data/pretrained/words.txt')
    group_embeddings = get_homophone_group_embeddings()
    homophone_chars = list(group_embeddings[0].keys())
    homophone_embeddings = np.array(list(group_embeddings[0].values()))
    baseline_chars = []
    baseline_embeddings = []

    for i in range(1, len(homophone_chars)):
        for j in range(1, 100):
            h2_index = freq_list.index(homophone_chars[i])
            if h2_index + j < len(freq_list) and freq_list[h2_index + j] not in homophone_chars:
                similar_frequency_char = freq_list[h2_index + j]
                break
            elif h2_index - j >= 0 and freq_list[h2_index - j] not in homophone_chars:
                similar_frequency_char = freq_list[h2_index - j]
                break

        baseline_chars.append(similar_frequency_char)
        baseline_embeddings.append(word2vec(similar_frequency_char).detach().numpy())

    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    homophone_embeddings_2d = tsne.fit_transform(homophone_embeddings)
    baseline_embeddings_2d = tsne.fit_transform(np.array(baseline_embeddings))

    # Plot the embeddings
    matplotlib.rcParams.update(
        {
            'font.family': 'Heiti TC',
            'axes.unicode_minus': False
        }
    )
    plt.figure(figsize=(10, 8))
    plt.scatter(homophone_embeddings_2d[:, 0], homophone_embeddings_2d[:, 1], alpha=0.7, color='blue')
    plt.scatter(baseline_embeddings_2d[:, 0], baseline_embeddings_2d[:, 1], alpha=0.7, color='red')

    # Annotate points with characters
    for i, char in enumerate(homophone_chars):
        plt.annotate(char, (homophone_embeddings_2d[i, 0], homophone_embeddings_2d[i, 1]), fontsize=8)

    for i, char in enumerate(baseline_chars):
        plt.annotate(char, (baseline_embeddings_2d[i, 0], baseline_embeddings_2d[i, 1]), fontsize=8)

    plt.title('t-SNE Visualization of Character Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
