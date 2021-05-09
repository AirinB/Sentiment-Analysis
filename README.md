# Sentiment-Analysis

Sentiment Analysi is the process of computationally defining and categorizing opinions expressed in a piece of text, especially to decide if the writer's attitude toward a specific subject  is positive or negative.

# Dataset

We used the [IMDB](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format), which is categorised 0 - negative or 1 - positive. Also we checked multiple resorces such as [this](https://www.analyticsvidhya.com/blog/2020/03/pretrained-word-embeddings-nlp/) one or [this](https://medium.com/analytics-vidhya/sentiment-analysis-for-text-with-deep-learning-2f0a0c6472b5) in order to gain an overview how other people solve this problem.


# How we built it

### Bag of Words

We used the `Bow Classifier` for our Model. For it we created a class BoWClassifier that inhetirts the `torch nn`module. Also we implimented the early stopping in order to avoid overffiting and reduce the waiting time to train the model.


#### __Results__
 The dev accuracy in the last epoch (because of the early stopping it is `87.92`% and the dev loss it is `0.418` 

<img width="430" alt="Screenshot 2021-05-09 at 16 49 51" src="https://user-images.githubusercontent.com/27647952/117576518-94c77b80-b0e6-11eb-93b9-af19a8a8bab9.png">

  Never the less we can see pretty big dicrepancy between dev and train loss and accuracy so this might indicate that the model is a little overffit. SInce we provided a lot of data for the training we suspect that the problem is that this model is too simple. This is why we tried also to train on the `LSTM classifier`
<img width="1015" alt="Screenshot 2021-05-09 at 16 49 14" src="https://user-images.githubusercontent.com/27647952/117576485-7f525180-b0e6-11eb-8bb7-4a98c30e7303.png">







### LSTM sequence classifier model with pretrained embeddings 


## Results



# Challenges we ran into



# Accomplishments that we're proud of



# What we learned

