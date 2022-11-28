#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jose F Naranjo
# California State University - Stanislaus
# Sentiment Analysis using a Natural Language Processing neural network (transformers library, BERT neural network)

# As a business owner, content or product creator, or even a manager, have you ever wondered what your consumers really think about your products?
# Or more precisely, how they FEEL about your product?


# In[2]:


# Imports:
#!pip install transformers requests beautifulsoup4 pandas numpy
#!pip install matplotlib

from transformers import AutoTokenizer, AutoModelForSequenceClassification # <-- 1) Pass through a string and convert into a sequence of numbers to pass to npl model. 2) Architecture from transformers to load in npl model
from bs4 import BeautifulSoup                                              # To traverse reviews from website.
import torch                                                               # Extract highest requence result.
import requests                                                            # To grab data from website.
import re                                                                  # RegEx function to extract specific comments.
import numpy as np                                                         # To convert reviews into numpy arrays.
import pandas as pd                                                        # To get reviews into data frame.
import matplotlib.pyplot as plt                                            # Needed for graphing.


# In[3]:


# Instatiate model:
# NLP models for translation, Q&A, classification and generation.

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[4]:


# Test a made up review:
# Pass through string to tokenizer. Then, pass it through the model to get a classification.

review = tokenizer.encode('This movie was a complete waste of my time, absolutely terrible. Waste of money.', return_tensors='pt')


# In[5]:


# Review converted the string into a sequence of numbers:
review


# In[6]:


# Decode list. The original text can be seen below. I used 'review[0]' because you can't pass through a list of lists.
# The list at [0] is the list that is needed (internal list/string).
tokenizer.decode(review[0])


# In[37]:


# Encode and calculate sentiment:
# Test:

review = tokenizer.encode('This movie was great. Overall good movie', return_tensors='pt') # <-- Encoded string
result = model(review)
result   # <-- Sequence classifier output class. The output is an encoded list of score (the values < 1 still represent a rating of 1.)


# In[38]:


# The values inside the tensor/array represent the probability of a class (position) being the sentiment.
# Below you can take a clearer look of the tensor results:
result.logits


# In[39]:


sentiment = int(torch.argmax(result.logits)) + 1 # Gets the highest value result (which would be position "1" from the output above). The highest value represents what position represents the sentiment.
                                                 # The higher the value, the better. Likewise, the lower the worst the sentiment is.
print(f'Sentiment value:', sentiment)    


# In[10]:


# Real reviews:
# To extract real reviews from a website that has a review section, RegEx will be needed.

links = []
reviews =[]
results =[]

for i in range(0, 13):
    page = i * 10
    url = 'https://www.yelp.com/biz/in-shape-manteca-manteca-3?osq=Gyms&start=' + str(page) + '&sort_by=date_asc'
    links.append(url)
links.reverse()
print(links)

for i in range(len(links)):
    url = links[i]
    extractor = requests.get(url)
    parser = BeautifulSoup(extractor.text, 'html.parser')

    # Pass RegEx (full-stop, asterisk, comment, fullstop, asterisk) through parser (BeautifulSoup):
    regex = re.compile('.*comment*')          # <-- Looking for anything that has a comment within class (review, html file)

    results += parser.find_all('p', {'class':regex}) # 'p' means paragraph.. <p class="comment__"...>
reviews = [result.text for result in results]


# In[11]:


# Text gotten with 'extractor' is passed to beautifulsoup 'parser', and parsed through using 'html.parser'.
# Test:
extractor.text


# In[12]:


# Test results:
results[0].text


# In[13]:


# Load reviews into dataframe and score:
# Scrape each and every review and score them!

dataf = pd.DataFrame(np.array(reviews), columns=['Review']) # <-- Makes it easier to go through the reviews and process them.
dataf

# In[14]:


# Test dataframe:

dataf.head()


# In[15]:


dataf.tail()


# In[22]:


dataf['Review'].iloc[5]


# In[23]:


# Using the functions previously implemented, contruct a method to pass through each review and rate it: 

def sentiment_analysis(review):
    line = tokenizer.encode(review, return_tensors='pt')
    result = model(line)
    return int(torch.argmax(result.logits)) + 1


# In[24]:


# Test the function:

sentiment_analysis(dataf['Review'].iloc[0])


# In[19]:


# As a final step, use lambda function to go through all the reviews in the dataframe, and store them in 'review' column:
# NLP pipeline has a limit of how much text you can pass through it. It will take the first 1024 tokens of each review.

dataf['Score (1 - 5)'] = dataf['Review'].apply(lambda x: sentiment_analysis(x[:1024]))
dataf


# In[20]:


plt.figure(figsize = (30, 15))
plt.title("Sentiment Analysis (on a scale from 1 to 5)", fontsize = 20)
plt.plot(dataf['Score (1 - 5)'], color = 'red', marker = "o", label = "Sentiment score")
plt.legend(loc = "lower right", fontsize = 12)
#plt.xticks(range(0, len(dataf['Review'].apply(lambda x: sentiment_analysis(x[:1024])))))
plt.xlabel("Review #", fontsize = 16)
plt.ylabel("Score", fontsize = 16)
plt.grid(True)
plt.show()
