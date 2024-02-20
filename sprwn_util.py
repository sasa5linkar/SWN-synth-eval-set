import os
import joblib
from dotenv import load_dotenv
from srpskiwn import srpskiwordnet
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Retrieve file root and file name from environment variables
file_root = os.getenv('srpwn_root')
file_name = os.getenv('srpwn_file')

# Create a new instance of the srpskiwordnet class
# This is a global variable and can be imported in other scripts
srpwn = srpskiwordnet.SrbWordNetReader(file_root, file_name)
lexicon = os.getenv('srpwn_lexicon')
srpwn.load_lexicon(lexicon)

def load_polarity_lexicon():
    """
    Load the Serbian polarity lexicon into a tuple of lists.

    Returns:

    tuple: A tuple of lists containing positive and negative words.

    note: The lexicon is loaded from the environment variable srp_pol
    it is cvs file with three columns: word, pos, neg, separated by semicolon
    there is no header in file 
    pos and neg are 1 if word is positive or negative, missing value is 0
    there are no objective words in the lexicon
    example:

    "agresivan";;1
    "akademija";1;
    "alkoholizam";;1
    "anarhist";;1
    "anatema";;1
    """
    #load file path from environment variable
    file_path = os.getenv('srp_pol')
    #open file
    with open(file_path, 'r', encoding="utf-8-sig") as file:
        #read file
        lines = file.readlines()
        #split lines by semicolon
        words = [line.split(';') for line in lines]
        #add to posive and negative list words that are positive or negative, and remove new line character
        #from negative and quote from word
        positive = [word[0].replace('"', '').replace('\n', '') for word in words if word[1] == '1']
        negative = [word[0].replace('"', '').replace('\n', '') for word in words if word[2].strip() == '1']
        return (positive, negative)
    
def check_polarity_of_synset_less_than_eq (synset: srpskiwordnet.SrbSynset, POS, NEG):
    """Checks if polarity of the synset is less or equal to POS and NEG.

    Args:
    synset: srpskiwordnet.SrbSynset
        Synset to be checked.
    POS: float
        Positive polarity. Between 0 and 1.
    NEG: float
        Negative polarity. Between 0 and 1.

    Returns:
    bool: True if synset polarity is less or equal to POS and NEG, False otherwise.

    """
    #get synset polarity
    pos, neg = synset._sentiment
    #check if synset polarity is less or equal to POS and NEG
    return pos <= POS and neg <= NEG

def find_irregular_synsets_under_threshold(threshold):
    """
    Find all synsets that are marked as polarized in Serbian polarity lexicon but
    have polarity less than threshold in Serbian WordNet, which is copied from SentiWordNet.
    
    Args:
    threshold (float): Threshold for polarity. Between 0 and 1.
    
    Returns:
    tuple: A tuple of lists containing positive and negative synsets that satisfy the condition.
    """
    # Load positive and negative words from lexicon
    positive, negative = load_polarity_lexicon()

    # Define a function to find synsets under the threshold


    def find_synsets(words):
        synsets_list = []
        for word in tqdm(words, desc="Processing words"):
            synsets = srpwn.synsets(word)
            synsets_list.extend([synset for synset in synsets if check_polarity_of_synset_less_than_eq(synset, threshold, threshold)])
        return synsets_list

    # Find positive and negative synsets under the threshold
    positive_synsets = find_synsets(positive)
    negative_synsets = find_synsets(negative)

    return positive_synsets, negative_synsets                

def save_synsets_to_file(synsets, file):
    """
    Save synsets to a file.

    Args:
    synsets (list): List of synsets to be saved.
    file (str): File name to save synsets to.
    """
    joblib.dump(synsets, file)

def load_synsets_from_file(file):
    """
    Load synsets from a file.

    Args:
    file (str): File name to load synsets from.

    Returns:
    list: List of synsets loaded from the file.
    """
    return joblib.load(file)

def convert_synset_to_dict(synset: srpskiwordnet.SrbSynset, lexicon_polarity):
    """
    Convert synset to a dictionary.

    Args:
    synset (srpskiwordnet.SrbSynset): Synset to be converted.

    Returns:
    dict: Dictionary representation of the synset.

    Notes:
    The dictionary contains the following
    keys: "ILI", "definition", "lemma_names", "sentiment_SWN"
    ILI is Iner-Lingual Index, definition is definition of the synset,
    deinition je dictionary definition of the synset, lemma_names is list of words that are synonyms for the synset
    sentiment_SWN is tuple of floats, repsesing polarity directly mapped from SentiWordNet
    """
    return {
        "ILI": synset._ID,
        "definition": synset._definition,
        "lemma_names": synset._lemma_names,
        "sentiment_SWN": synset._sentiment,
        "sentiment_lexicon": lexicon_polarity
    }

def convert_synsets_to_dicts(synsets, lexicon_polarity):
    """
    Convert synsets to dictionaries.

    Args:
    synsets (list): List of synsets to be converted.

    Returns:
    list: List of dictionaries representing the synsets.
    """
    return [convert_synset_to_dict(synset, lexicon_polarity) for synset in synsets]

def add_converted_synsets_to_list(synsets, list_of_dicts, lexicon_polarity):
    """
    Add converted synsets to a list of dictionaries.

    Args:
    synsets (list): List of synsets to be converted.
    list_of_dicts (list): List of dictionaries to add the converted synsets to.
    """
    list_of_dicts.extend(convert_synsets_to_dicts(synsets, lexicon_polarity))

def save_converted_synsets_to_file(list_of_dicts, file):
    """
    Save converted synsets to a file.

    Args:
    list_of_dicts (list): List of dictionaries to save.
    file (str): File name to save the list of dictionaries to.
    """
    joblib.dump(list_of_dicts, file)
def load_converted_synsets_from_file(file):
    """
    Load converted synsets from a file.

    Args:
    file (str): File name to load the list of dictionaries from.

    Returns:
    list: List of dictionaries loaded from the file.
    """
    return joblib.load(file)

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

    positive, negative = load_polarity_lexicon()
    #just print first 10 words from positive and negative list
    print(positive[:10])
    print(negative[:10])

    # Find synsets with polarity equal to or less than 0
    positive_synsets, negative_synsets = find_irregular_synsets_under_threshold(0)
    # Print the first 10 synsets, for each synset print its definition, lemma and polarity
    
    print("Positive synsets:")
    for synset in positive_synsets[:20]:
        print(synset._definition, synset._lemma_names, synset._sentiment)
        print("-------------------")
    
    print("*"*50)
    print("Negative synsets:")

    for synset in negative_synsets[:20]:
        print(synset._definition, synset._lemma_names, synset._sentiment)
        print("-------------------")

    #prnt number of synsets
    print("Number of irregular positive", len(positive_synsets))
    print("Number of irregular negative", len(negative_synsets))
    #save synsets to file
    save_synsets_to_file(positive_synsets, "positive_synsets")
    save_synsets_to_file(negative_synsets, "negative_synsets")

    list_of_dicts = convert_synsets_to_dicts(positive_synsets, "positive")
    add_converted_synsets_to_list(negative_synsets, list_of_dicts, "negative")
    save_converted_synsets_to_file(list_of_dicts, "converted_synsets")



if __name__ == '__main__':
    test()