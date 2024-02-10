import os
from dotenv import load_dotenv
from srpskiwn import srpskiwordnet

# Load environment variables
load_dotenv()

# Retrieve file root and file name from environment variables
file_root = os.getenv('srpwn_root')
file_name = os.getenv('srpwn_file')

# Create a new instance of the srpskiwordnet class
# This is a global variable and can be imported in other scripts
srpwn = srpskiwordnet.SrbWordNetReader(file_root, file_name)

def get_def_by_word(word):
    """
    Given a word, retrieve its definitions from the Serbian WordNet.

    Parameters:
    word (str): The word to find definitions for.

    Returns:
    list: A list of definitions for the given word.
    """
    synsets = srpwn.synsets(word)
    return [synset.definition() for synset in synsets]

def get_def_by_words(words):
    """
    Given a list of words, retrieve their definitions from the Serbian WordNet.

    Parameters:
    words (list): The words to find definitions for.

    Returns:
    list: A list of definitions for the given words.
    """
    return [definition for word in words for definition in get_def_by_word(word)]

def test():
    """
    Test function to print definitions of specific words.
    """
    print(get_def_by_word('sreća'))
    print(get_def_by_word('bol'))

    list_of_words = ['sreća', 'bol']
    print(get_def_by_words(list_of_words))

if __name__ == '__main__':
    test()