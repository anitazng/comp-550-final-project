import hanlp
import torch
from wordfreq import word_frequency as wf
import csv
import pandas as pd
from pypinyin import lazy_pinyin, Style
from collections import defaultdict

#word embedding stuff
word2vec = hanlp.load(hanlp.pretrained.word2vec.RADICAL_CHAR_EMBEDDING_100)
cos = torch.nn.CosineSimilarity(dim=0)

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

def group_pseudo_homophones_from_file(charfile):
    '''
    Reads in file with characters.
    Group pseudo-homophones (same pinyin, not necessarily tone) from characters.
    Returns dictionary with key:pinyin and value:list of characters that correspond to that pinyin.
    '''
    homophone_groups = defaultdict(list)
    with open(charfile, 'r') as file:
            lines = file.readlines()
            chars = [line[0] for line in lines]
    
    for char in chars:
        pinyin = lazy_pinyin(char, style=Style.NORMAL) # do not take tone into account
        homophone_groups["".join(pinyin)].append(char)
    
    return homophone_groups
     

def get_frequency_list(charfile):
    '''
    Returns list of characters in order of decreasing frequency according to wordfreq.
    '''
    with open(charfile, 'r') as file:
            lines = file.readlines()
            chars = [line[0] for line in lines]

    return chars

def compute_distance(homophone_groups, freq_list, output):
    '''
    Prints average baseline similary, average homophone group similarity, and difference between the two.
    Creates new data file with homophone and homophone or homophone and similar frequency char entries.
    '''
    baseline_similarity = 0
    baseline_counter = 0
    homophone_similarity = 0
    homophone_counter = 0
    with open(file=output, mode="w") as filename:
        writer = csv.writer(filename)
        for homophones in homophone_groups.values():
            homophones = set(homophones)
            if len(homophones) > 1:
                for h1 in homophones: # compute average homophone pair distance and baseline distance for each homophone
                    for h2 in homophones:
                        if h1 != h2:
                            # compute baseline similarities
                            for i in range(1, 100):
                                h2_index = freq_list.index(h2)
                                if h2_index + i < len(freq_list) and freq_list[h2_index + i] not in homophones:
                                    similar_frequency_char = freq_list[h2_index + i]
                                    break
                                elif h2_index - i >= 0 and freq_list[h2_index - i] not in homophones:
                                    similar_frequency_char = freq_list[h2_index - i]
                                    break
                  
                            #writes baseline comparison to data file
                            baseline_similarity += cos(word2vec(h1), word2vec(similar_frequency_char))
                            row = [h1, similar_frequency_char, cos(word2vec(h1), word2vec(similar_frequency_char)).item(), 1]
                            writer.writerow(row)
                            baseline_counter += 1

                            #writes homophone comparison to data file
                            homophone_similarity += cos(word2vec(h1),word2vec(h2))
                            row = [h1, h2, cos(word2vec(h1), word2vec(h2)).item(), 0]
                            writer.writerow(row)
                            homophone_counter += 1
                        

    baseline_average = baseline_similarity / baseline_counter
    homophone_average = homophone_similarity / homophone_counter
    print(f'Average Baseline Similarity: {baseline_average}')
    print(f'Average Homophone Group Similarity: {homophone_average}')
    print(f'Difference Between Average Baseline and Homophone Similarities: {baseline_average - homophone_average}')

if __name__ == "__main__":
    true_homophone_groups = group_true_homophones_from_file('data/pretrained/words.txt')
    pseudo_homophone_groups = group_pseudo_homophones_from_file('data/pretrained/words.txt')
    freq_list = get_frequency_list('data/pretrained/words.txt')

    compute_distance(true_homophone_groups, freq_list, 'data/pretrained/true_homophone_output.csv')
    compute_distance(pseudo_homophone_groups, freq_list, 'data/pretrained/pseudo_homophone_output.csv')