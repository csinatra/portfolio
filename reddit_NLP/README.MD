# NLP Analysis of Political Subreddits

## Problem Statement
The purpose of this project is to use Natural Language Processing techniques to identify linguistic differences between closely related but ideologically distinct political subgroups. In this case, the groups of interest are Conservatives and Libertarians. Creating accurate classification models has application in assessing journalistic sources as well as optimizing communications to create a deeper connection with the target audience.

## Executive Summary
Our model classified our data with 72.4% accuracy, showing a positive proof of concept. The best model was multinomial naive bayes, a relatively simple model that suggests there is room for significant improvement as we move toward more sophisticated models. The baseline accuracy was 53.8%

## Background
As access to information becomes more distributed across web sources the ability to verify journalistic integrity becomes a pressing concern. This is made clear by ongoing issues surrounding social media platforms' promotion of 'fake news' designed to influence political outcomes as well as the broader public conversation around bias in media.

This project aims to identify semantic and syntactical markers associated with two similar but distinct political subgroups: Conservatives and Libertarians. Understanding the ideological underpinnings that define communication dynamics within groups can be applied by modeling baseline tone for a given source and alerting audiences to significantly divergent texts. It may also be used for identifying bots or other inauthentic sources which may attempt to influence group sentiment.

Outside of propaganda detection, accurate modeling may also be used for scoring articles or press releases for relevance among the target audience. By scoring words that hold weight within a group the communication may be optimized for relatability, impact, scope, and demographics.

A text corpus will be built by querying Reddit and compiling comments from the Conservative and Libertarian subreddits that can be used for Supervised Learning using various classification models. The Pushshift API will be used to collect 100,000 samples from each subreddit and the data will be cleaned using Regex and word vectorization prior to Natural Language Processing (NLP). The clean data will be divided into training and validation sets before being fit by a given model before predicting classifications using the validation dataset. Models will include CART methods, Support Vector Machines (SVM), K-Nearest Neighbors / Latent Semantic Analysis (KNN / LSA), and others before determining a final production model.

## Findings
During EDA we found that Libertarians generally had higher word and character counts for a given document compared with Conservatives. While Count Vectorization produced similar top word lists, Tfidf Vectorization was able to identify words that had a stronger association with each groups individual ideology. As such, Tfidf data was used for modeling.

Our baseline prediction was 53.8% accuracy by predicting the positive class for all samples.

The Random Forest model showed strong performance with the training data however it performed poorly on the validation data, showing signs of overfitting. The predictive accuracy on untrained data was only 70.5%, less than the naive bayes model. It's likely this is was a result of using a small number of trees for fitting, and considering a larger number for each split would produce a more generalizable result.

The Multinomial Naive Bayes model performed consistently between the training and validation datasets, predicting classes with 72.4% accuracy. Considering the simplicity fo the model and that the model was fit to Term Frequency / Inverse Document Frequency vectorized data this is a strong increase over the baseline model.

## Recommendations
- Fit additional Models

- Use GridSearch to optimize over a larger space of hyperparameters

- More aggressive data cleaning to streamline computations and reduce compute time.

## Conclusions
This preliminary investigation showed strong positive results and set the groundwork for further investigations. Continuing with a wider group of model algorithms as well as allocating more compute to tuning our models is likely to produce a generalizable model that predicts with a much higher accuracy.For initial studies, the Multinomial Naive Bayes may serve as the minimum viable product to put this classification system into production.
