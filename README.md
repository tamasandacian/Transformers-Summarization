# Transformers-Summarization

Transformers-Summarization is a Python-based library using transformers meant to help generate abstractive summaries from an input given text. It can be used for summarizing long documents such as (e.g blog, news). Examples of summarization methods include: T5, BART, GPT-2, XLM.

## Setup
```
1. Clone repository

2. install conda library 
   pip3 install conda

3. create conda environment
   conda create --name sum
   conda activate sum
   
4. install required libraries
   conda install flask
   conda install pandas
   conda install numpy
   conda install pytorch
   conda install transformers

```

```python

from summarizer import Summarizer

text = """
       Machine learning (ML) is the study of computer algorithms that improve automatically through experience. 
       It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model 
       based on sample data, known as "training data", in order to make predictions or decisions without being explicitly 
       programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering 
       and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks.
       """

s = Summarizer(method='T5', pretrained='t5-large')
pred = s.summarize(text)

print(pred)

'''
    {
      "summary": "machine learning (ML) is the study of computer algorithms that improve automatically through experience. ML algorithms
                  build a mathematical model based on sample data in order to make predictions or decisions without being explicitly programmed to do so. they are 
                  used in wide variety of applications, such as email filtering and computer vision",
      "message": "successful"
    }
 '''
```
