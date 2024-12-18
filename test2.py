from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from pickle import load
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
# model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
def predict_answer(context, question):
    tokenizer = load(open("distilbert-base-uncased-tokenizer.pickle",'rb'))
    model = load(open("distilbert-base-uncased-distilled-squad.pickle", 'rb'))
    # context = "The US has passed the peak on new coronavirus cases, " \
    #         "President Donald Trump said and predicted that some states would reopen this month. " \
    #         "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."

    # question = "What was President Donald Trump's prediction?"

    encoding = tokenizer.encode_plus(question, context)


    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    return answer_tokens


# print ("\nQuestion ",question)
# print ("\nAnswer Tokens: ")
# print (predict_answer(""))