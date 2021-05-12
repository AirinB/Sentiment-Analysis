# Sentiment-Analysis

Sentiment Analysi is the process of computationally defining and categorizing opinions expressed in a piece of text, especially to decide if the writer's attitude toward a specific subject  is positive or negative.

# Dataset

We used the [IMDB](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format), which is categorised 0 - negative or 1 - positive. Also we checked multiple resorces such as [this](https://www.analyticsvidhya.com/blog/2020/03/pretrained-word-embeddings-nlp/) one or [this](https://medium.com/analytics-vidhya/sentiment-analysis-for-text-with-deep-learning-2f0a0c6472b5) in order to gain an overview how other people solve this problem.


# How we built it

### Bag of Words

We used the `Bow Classifier` for our Model. For it we created a class BoWClassifier that inhetirts the `torch nn.Module` module. Also we implimented the early stopping in order to avoid overffiting and reduce the waiting time to train the model.


#### __Results__
 The dev accuracy in the last epoch (because of the early stopping it is `87.92`% and the dev loss it is `0.418` 

<img width="430" alt="Screenshot 2021-05-09 at 16 49 51" src="https://user-images.githubusercontent.com/27647952/117576518-94c77b80-b0e6-11eb-93b9-af19a8a8bab9.png">

  Never the less we can see pretty big dicrepancy between dev and train loss and accuracy so this might indicate that the model is a little overffit. SInce we provided a lot of data for the training we suspect that the problem is that this model is too simple. This is why we tried also to train on the `LSTM classifier`
<img width="1015" alt="Screenshot 2021-05-09 at 16 49 14" src="https://user-images.githubusercontent.com/27647952/117576485-7f525180-b0e6-11eb-8bb7-4a98c30e7303.png">



#### Challenges we ran into

- [x] Hard to improve the accuracy
- [x] Memory issues



### LSTM sequence classifier model with pretrained embeddings 

For the second model we used Sequence Classification with `LSTM Recurrent Neural Networks`. For it we also created a separate class `LSTMClassifier` which also inherits the `nn.Module` and we tokenized the sentances using the `Tokenizer`module from `keras`. Also we introduced downsampling in order to speed up the time and to check if this will affect our neural network.

## Glove Embeddings 

Global Vectors for Word Representation, or GloVe, is an “unsupervised learning algorithm for obtaining vector representations for words.” 
Training is performed on aggregated global word-word co-occurrence statistics from a corpus. It is developed by Stanford.
We used the glove.6B.100d embeddings for the model.

#### __Results__

We got only 46% accuracy for dev and 0.824 for the loss which we consider very bad results. 

<img width="358" alt="Screenshot 2021-05-09 at 17 03 58" src="https://user-images.githubusercontent.com/27647952/117577909-208fd680-b0ec-11eb-8e47-9ec00055a035.png">
<img width="984" alt="Screenshot 2021-05-09 at 17 04 16" src="https://user-images.githubusercontent.com/27647952/117577920-30a7b600-b0ec-11eb-84bd-abbe4c6178e0.png">

**Test accuracy**
51.5%

We decided to use other metrics to understand the poor results

**Precision**

When evaluating the sentiment (positive, negative, neutral) of a given text document, the baseline of precision lies around 
80-85% . This is the baseline we try to meet or beat when we're training a sentiment scoring system.
Test Precision : 75.9%
Which is lower than the baseline.

**Confusion Matrix**

A confusion matrix is a method of visualizing classification results.
Confusion matrix will show you if your predictions match the reality and how do they match in more detail.

![image](https://user-images.githubusercontent.com/41289743/118036377-1ee33e80-b38a-11eb-8e63-45d05be6421b.png)


The Confusion matrix helps us understand how many correct prediction does the model make

![image](https://user-images.githubusercontent.com/41289743/118035783-50a7d580-b389-11eb-9b80-7030ea5982c0.png)




#### Challenges we ran into

- [x] Random kernel stopping when using `pandas`
- [x] Memory issues
- [x] Doing the Sanity check the notebook just stops because it requires too much memory so we commented out this part


# What we learned

- [x] use `Python`libraries to create models
- [x] impove the accuracy by testung models with different parametes
- [x] solving an nlp problme from scratch
- [x] overcoming memory issues by decreasing the dataset or batch size  
