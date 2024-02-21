import json
import os
import random
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import sprwn_util
from sprwn_util import get_def_by_words, srpwn, load_converted_synsets_from_file
import joblib
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from tqdm import tqdm
import logging
import warnings
import pandas as pd

# Ignore warnings
# Configure logging to write warnings to a log file
logging.basicConfig(filename='warnings.log', level=logging.WARNING)

# Redirect warnings to the logging system
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_prompt_from_file(file):
    """Loads prompt from file.

    Args:
    file: str
        File name.

    Returns:
    PromptTemplate: Prompt from file.

    """
    with open(file, 'r', encoding="utf-8") as f:
        return PromptTemplate.from_template(f.read())

class SentimentAnaliser:
    """Class for sentiment analisys of the text. It uses HuggingFace pipeline for text generation.
    
    Attributes:
    model_name: str
        Name of the model used for sentiment analisys.
    prompt: str
        Prompt for the model.
    lmm: HuggingFacePipeline
        Language model microservice.
    chain: LangChain
        Language chain.
    asnwer: str
        Answer from the model.    

    """
    
    task="text-generation"
    pipeline_kwargs={"max_new_tokens": 5}
    def __init__(self, model_name, prompt, max_new_tokens=5):
        """Initializes SentimentAnaliser with model name and prompt.

        Args:
        model_name: str
            Name of the model used for sentiment analisys.
        prompt: str
            Prompt for the model.

        """
        if max_new_tokens!=5:
            self.pipeline_kwargs["max_new_tokens"] = max_new_tokens
        
        self.prompt = prompt
        self.lmm = HuggingFacePipeline.from_model_id(model_id= model_name, task = SentimentAnaliser.task, 
                                                     pipeline_kwargs=SentimentAnaliser.pipeline_kwargs, 
                                                     device=0)
        self.chain = self.prompt | self.lmm
    
    def analyze(self, text):
        """Analyzes the text and sets the answer.

        Args:
            text (str): Text to be analyzed.

        Returns:
            dict: The result of the analysis.

        """
        return self.clean_answer(self.chain.invoke({"text": text}))
    def clean_answer(self, text):
        """Cleans the answer. Removes dots and colons.
        Sometimes we get the answer from LLM with dots and colons at the end. This function removes them.

        """
        text = text.replace(".", "")
        text = text.replace(":", "")
        text = text.replace("-", "")
        return text.strip()

def analyse_list (list_text, sentimentAnaliser: SentimentAnaliser):
    """Analises the list of texts and return list of tuples (text, answer).

    Args:
    list_text: list
        List of texts to be analised.
    sentimentAnaliser: SentimentAnaliser
        SentimentAnaliser object used for analisys.
    
    Returns:
    list: List of tuples (text, answer).

    """
    

    return [(text, sentimentAnaliser.analyze(text)) for text in tqdm(list_text, desc="Analysing sentiments")]

def save_sentment_list_as_cvs(list_text, sentimentAnaliser: SentimentAnaliser, file):
    """Analises the list of texts and save it as cvs file, using ¦ as separator.

    Args:
    list_text: list
        List of texts to be analised.
    sentimentAnaliser: SentimentAnaliser
        SentimentAnaliser object used for analisys.
    file: str

    """
    list_result = analyse_list(list_text, sentimentAnaliser)
    with open(file, 'w',encoding="utf-8") as f:
        f.write("Text;Sentiment\n")
        for text, result in list_result:
            f.write(f"{text}¦{result}\n")
    


# Test prompt for sentiment analysis in English
test_prompt = """
    As a sentiment analysis expert, analyze the following Serbian text and determine its sentiment. 
    The sentiment should be classified strictly as "positive", "negative", or "objective". No other responses will be accepted.
    Text: {text}
    Sentiment: 
"""
# Test prompt for sentiment analysis in Serbian

test_prompt_sr = """
    Kao ekspert za analizu sentimenta, analizirajte sledeći tekst na srpskom jeziku i odredite njegov sentiment. 
    Sentiment treba da bude striktno klasifikovan kao "pozitivan", "negativan", ili "objektivan". Nijedan drugi odgovor neće biti prihvaćen.
    Tekst: {text}
    Sentiment: 
"""

test_prompt_positive_sr = """
    Kao ekspert za analizu sentimenta, analizirajte sledeći tekst na srpskom jeziku i odredite da ima pozitivan sentiment.
    Sentiment treba da bude striktno klasifikovan kao "nije pozitivan", "slabo pozitivan", "umereno pozitivan", "veoma pozitivan", ili "ekstremno pozitivan". Nijedan drugi odgovor neće biti prihvaćen. 
    Nijedan drugi odgovor neće biti prihvaćen.  
    Tekst: {text}
    Pozitivan sentiment:
"""

test_prompt_positive_sr2 = """
    Kao stručnjak za analizu sentimenta, vaš zadatak je da pažljivo procenite dati tekst na srpskom jeziku. 
    Na osnovu vaše analize, klasifikujte sentiment teksta koristeći striktno definisane kategorije. 
    Ove kategorije uključuju: 'nije pozitivan', 'slabo pozitivan', 'umereno pozitivan', 'veoma pozitivan', i 'ekstremno pozitivan'. 
    Važno je naglasiti da su ovo jedine prihvatljive kategorije za klasifikaciju. 
    Molimo vas da se držite ovih smernica kako biste osigurali tačnost i konsistentnost u analizi sentimenta. 
    Tekst za analizu: {text}. 
    Očekujemo da vaša analiza rezultira određivanjem jedne od navedenih kategorija sentimenta.
    Kakav je sentiment teksta?
    Kategorija sentimenta:
    """

def test_old():
    word_list = ["sreća", "bol", "radost", "tuga", "ljubav", "mržnja", "sloboda", "zatvor", "život", "smrt"]

    text_list = get_def_by_words(word_list)
    #remove empty strings
    text_list = list(filter(None, text_list))


    print("Analised texts saved as sentiment.csv")
    list_synsets = load_converted_synsets_from_file("converted_synsets")
    #get random sample of 500 synsets from list
    sample_synsets = random.sample(list_synsets, 500)
    

    for synset in tqdm(sample_synsets, desc="Processing synsets"):
        synset["sentiment_sa"] = sa.analyze(synset["definition"])
    with open("sample_synsets2.json", 'w',encoding="utf-8") as f:
        f.write(json.dumps(sample_synsets, indent=4)) 
    dataset = pd.DataFrame(sample_synsets)
    dataset.to_csv("sample_synsets2.csv", sep="¦", index=False)

def test():
    word_list = ["sreća", "bol", "radost", "tuga", "ljubav", "mržnja", "sloboda", "zatvor", "život", "smrt"]
    text_list = get_def_by_words(word_list)
    #remove empty strings
    text_list = list(filter(None, text_list))
    sa = SentimentAnaliser("mistralai/Mistral-7B-Instruct-v0.2", PromptTemplate.from_template(test_prompt_positive_sr), max_new_tokens=10)
    save_sentment_list_as_cvs(text_list, sa, "sentiment_positive.csv")
    sa = SentimentAnaliser("mistralai/Mistral-7B-Instruct-v0.2", PromptTemplate.from_template(test_prompt_positive_sr2), max_new_tokens=10)
    save_sentment_list_as_cvs(text_list, sa, "sentiment_positive2.csv")
if __name__ == "__main__":
    test()
