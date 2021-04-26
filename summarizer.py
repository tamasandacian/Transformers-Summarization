import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
from utility import Utility
import logging

T5_PRETRAINED_MODELS = {
    't5-small', 't5-base', 't5-large'
}

BART_PRETRAINED_MODELS = {
    'facebook/bart-base','facebook/bart-large-cnn'
}

GPT2_PRETRAINED_MODELS = {
    'gpt2','gpt2-medium', 'gpt2-large', 'gpt2-xl'
}

XLM_PRETRAINED_MODELS = {
    'xlm-mlm-en-2048'
}

class Summarizer(object):
    """ Class to generate abstractive summarization from input text 

    :param method: method type T5 / BART / GPT-2 / XLM
    :param pretrained: pretrained model name
    :param min_length: min number of words in a document
    :param max_length: max number of words in a document
    :param skip_special_tokens: boolean flag to filter out BERT tags
    """
    def __init__(self, method='T5', pretrained='t5-base', min_length=10, max_length=512, num_beams=4,
                 no_repeat_ngram_size=2, skip_special_tokens=True, return_tensors='pt', early_stopping=True):
        
        self.method = method
        self.min_length = min_length
        self.max_length = max_length
        self.pretrained = pretrained
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.skip_special_tokens = skip_special_tokens
        self.return_tensors = return_tensors
        self.early_stopping = early_stopping
        self.model, self.tokenizer = self.load_model_tokenizer(pretrained)

    def load_model_tokenizer(self, pretrained):
        """ Load transformer model and tokenizer for given pre-trained name 
        
        :param pretrained: pre-trained name
        :return: model, tokenizer
        """
        
        model = None
        tokenizer = None
        
        if self.method == "T5":
            if pretrained in T5_PRETRAINED_MODELS:
                model = T5ForConditionalGeneration.from_pretrained(pretrained)
                tokenizer = T5Tokenizer.from_pretrained(pretrained)
        elif self.method == "BART":
            if pretrained in BART_PRETRAINED_MODELS:
                model = BartForConditionalGeneration.from_pretrained(pretrained)
                tokenizer = BartTokenizer.from_pretrained(pretrained)
        elif self.method == "GPT-2":
            if pretrained in GPT2_PRETRAINED_MODELS:
                model = GPT2LMHeadModel.from_pretrained(pretrained)
                model.config.max_length = self.max_length
                tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
        elif self.method == "XLM":
            if pretrained in XLM_PRETRAINED_MODELS:
                model = XLMWithLMHeadModel.from_pretrained(pretrained)
                model.config.max_length = self.max_length
                tokenizer = XLMTokenizer.from_pretrained(pretrained)
        else:
            pass

        return model, tokenizer

    def extract_summary(self, text):
        """ Extract summary from input text 
        
        :param text: input textual content
        :return: summary text
        """
        input_ids = self.tokenizer.encode(text, return_tensors=self.return_tensors)
        
        summary_ids = self.model.generate(
            input_ids, num_beams=self.num_beams, no_repeat_ngram_size=self.no_repeat_ngram_size, 
            min_length=self.min_length, max_length=self.max_length, early_stopping=self.early_stopping
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=self.skip_special_tokens)
        return summary

    def summarize(self, text):
        """ Generate abstractive summary from input text 
        
        :param text: content text
        :return: Python dictionary
        """
        
        try:

            data = dict()

            if text:
                
                doc_len = Utility.get_doc_length(text)

                if doc_len > self.min_length:

                    if doc_len <= self.max_length:
                    
                        summary = ""

                        if self.method in { "T5", "BART", "GPT-2", "XLM" }:
    
                            if self.model and self.tokenizer:
                            
                                if self.method == "T5":
                                    
                                    # Add 'summarize' prefix
                                    text = "summarize:" + text
                                    summary = self.extract_summary(text)

                                else:
                                    summary = self.extract_summary(text)

                                summary = Utility.remove_newlines(summary)
                                summary = Utility.remove_multiple_whitespaces(summary)

                                data["summary"] = summary
                                data["message"] = "successful"

                            else:
                                return "model not exist"
                        else:
                            return "method not exist"
                    else:
                        return "number of words exceded {} / {}".format(doc_len, self.max_length)
                else:
                    return "required at least {} words for summarization".format(self.min_length)
            else:
                return "required textual content"

            return data
        except Exception:
            logging.error('exception occured', exc_info=True)