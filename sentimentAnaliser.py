import os
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import sprwn_util
from sprwn_util import get_def_by_words, srpwn


def load_prompt_from_file(file):
    with open(file, 'r') as f:
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
    def __init__(self, model_name, prompt):
        """Initializes SentimentAnaliser with model name and prompt.

        Args:
        model_name: str
            Name of the model used for sentiment analisys.
        prompt: str
            Prompt for the model.

        """
        self.prompt = prompt
        self.lmm = HuggingFacePipeline.from_model_id(model_id= model_name, task = SentimentAnaliser.task, 
                                                     pipeline_kwargs=SentimentAnaliser.pipeline_kwargs, 
                                                     device=0)
        self.chain = self.prompt | self.lmm
    
    def analise(self, text):
        """Analises the text and sets the answer.

        Args:
        text: str
            Text to be analised.

        """

        self.asnwer = self.chain.invoke({"text": text})

test_prompt = """
    As a sentiment analysis expert, analyze the following Serbian text and determine its sentiment. 
    The sentiment should be classified strictly as "positive", "negative", or "objective". No other responses will be accepted.
    Text: {text}
    Sentiment: 
"""
test_prompt_sr = """
    Kao ekspert za analizu sentimenta, analizirajte sledeći tekst na srpskom jeziku i odredite njegov sentiment. 
    Sentiment treba da bude striktno klasifikovan kao "pozitivan", "negativan", ili "objektivan". Nijedan drugi odgovor neće biti prihvaćen.
    Tekst: {text}
    Sentiment: 
"""

def main():
    word_list = ["sreća", "bol", "radost", "tuga", "ljubav", "mržnja", "sloboda", "zatvor", "život", "smrt"]

    text_list = get_def_by_words(word_list)
    #remove empty strings
    text_list = list(filter(None, text_list))

    sa = SentimentAnaliser("mistralai/Mistral-7B-Instruct-v0.2", PromptTemplate.from_template(test_prompt_sr))
    for text in text_list:
        print(text)
        print("++++++++++++++++++++++++++++++++++")
        sa.analise(text)
        print(sa.asnwer)
        print("----------------------------------")

if __name__ == "__main__":
    main()
