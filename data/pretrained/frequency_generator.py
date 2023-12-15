# with open('character.vec.txt', 'r', encoding='utf-8') as input_file:
#     # Read lines from the input file
#     lines = input_file.readlines()

# # Open the output file for writing
# with open('output.txt', 'w', encoding='utf-8') as output_file:
#     # Iterate through each line in the input file
#     for line in lines:
#         # Extract the first character of the line
#         first_char = line[0]

#         # Check if the character is a Chinese character
#         if '\u4e00' <= first_char <= '\u9fff':
#             # Write the character to the output file
#             output_file.write(first_char + '\n')

import wordfreq

def get_chinese_character_frequency(input_file_path):
    # Open the input file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        # Read the first character from each line
        chinese_characters = [line[0] for line in input_file if line.strip() and '\u4e00' <= line[0] <= '\u9fff']

    # Use wordfreq to get the frequency of each Chinese character
    frequency_dict = {}
    for char in chinese_characters:
        frequency = wordfreq.word_frequency(char, 'zh')
        frequency_dict[char] = frequency

    return frequency_dict

def write_sorted_frequency_to_file(output_file_path, frequency_dict):
    # Sort characters by frequency in decreasing order
    sorted_characters = sorted(frequency_dict, key=frequency_dict.get, reverse=True)

    # Open the output file for writing
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Write Chinese character and its frequency to the output file
        for char in sorted_characters:
            frequency = frequency_dict[char]
            output_file.write(f'{char}: {frequency}\n')

# Replace 'input.txt' and 'output.txt' with your file names
input_file_path = 'output.txt'
output_file_path = 'words_withfreq.txt'

# Get Chinese character frequencies
frequency_dict = get_chinese_character_frequency(input_file_path)

# Write sorted frequencies to the output file in decreasing order
write_sorted_frequency_to_file(output_file_path, frequency_dict)