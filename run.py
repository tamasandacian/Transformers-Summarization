from summarizer import Summarizer


text = """
       Machine learning (ML) is the study of computer algorithms that improve automatically through experience. 
       It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model 
       based on sample data, known as "training data", in order to make predictions or decisions without being explicitly 
       programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering 
       and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks.
       """ 


############################# T5 ##################################
s = Summarizer(method='T5', pretrained='t5-large')
summary = s.summarize(text)
print("T5: ", summary)
###################################################################

############################# BART ################################ 
s = Summarizer(method='BART', pretrained='facebook/bart-large-cnn')
summary = s.summarize(text)
print("BART: ", summary)
###################################################################

############################# GPT-2 ###############################
s = Summarizer(method='GPT-2', pretrained='gpt2')
summary = s.summarize(text)
print("GPT-2: ", summary)
###################################################################

############################# XLM #################################
s = Summarizer(method='XLM', pretrained='xlm-mlm-en-2048')
summary = s.summarize(text)
print("XLM: ", summary)
###################################################################