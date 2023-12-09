

# 1 - JSON file modifier
import json
# Read the JSON file line by line
with open('data/dev.json', 'r', encoding='utf-8') as file:
    for line in file:
        # Load each line as a JSON object
        data = json.loads(line)

        # Extract and concatenate content values - CHANGE this to be the name of the entry that you want to extract, it's sometimes sentence and sometimes content
        sentences = [data['sentence']]

        # Write sentences to a new txt file
        with open('output_sentences.txt', 'a', encoding='utf-8') as output_file:
            for sentence in sentences:
                # Split content into sentences using '。' as the delimiter
                sentences_in_content = sentence.split('。')
                # Write each sentence to the output file
                for s in sentences_in_content:
                    output_file.write(s.strip() + '\n')

# 2 -  Gets rid of everything that isn't a Chinese character

import re

def remove_non_chinese(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as input_file:
        with open(output_file, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                # Use regular expression to keep only Chinese characters
                chinese_only = re.sub('[^\u4e00-\u9fff]', '', line)
                # Write the modified line to the output file if it is not a blank line
                if chinese_only != '':
                    output_file.write(chinese_only + '\n')

# Example usage
input_file_path = 'output_sentences.txt'
output_file_path = 'output_chinese_only.txt'

remove_non_chinese(input_file_path, output_file_path)