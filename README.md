# WiDS-2023 Je ne parle pas français

Hello Mentees! <br><br>
Welcome to Wids! <br>
In this project, we cover the essentials of Classical Machine Learning or the Pre-Deep Learning era, followed by an introduction to Deep Learning and Natural Language Processing, and finish by building a simple English to French Translator

## Week-wise distribution of content

1) Quick recap of Python, Introduction to pytorch and neural networks and NLP with word2vec, word embeddings, Distributional semantics
2) Introduction to pytorch and neural networks, RNNs and building a sentiment analyser
3) Text classification, Language Modeling Introducing attention(Transformer: Attention is All You Need) in encoder-decoders, building a transformer from scratch, Seq2Seq
4) Transfer learning, replacing pre-trained word embeddings in GPT and BERT, and building and training a translator in pytorch from scratch

## Checkpoints

* [ ] Basic assignment to assess your understanding of Python
* [ ] Building and training BERT and En-Fr Translator

## Week-1

### Resources

To get started with python: https://docs.python.org/3.11/tutorial/index.html

Python and Numpy Tutorial: https://cs231n.github.io/python-numpy-tutorial/

Below you will find links to various online resources to help you with this week's portion. Please go through them.

- [Python Basics](#python-basics)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [NumPy Basics](#numpy-basics)
  - [Optional Assignment](#optional-assignment)
- [Neural Networks](#neural-networks)
- [Pytorch](#pytorch)
- [Assignment](#assignment)

## Python Basics

You will mostly, if not exclusively, use Python throughout this course. You are not expected to have an in-depth knowledge of the language, but an understanding of the data types and basic data structures like lists (similar to arrays in C/C++) and dicts (similar to structs), the basic syntax for different types of loops, defining functions, reading from and writing to files, etc.  
For basic Python Tutorials refer to the links below -

- [**Quick video tutorial by Mosh**](https://www.youtube.com/watch?v=kqtD5dpn9C8) or if you feel like it, you can watch the [complete tutorial](https://www.youtube.com/watch?v=_uQrJ0TkZlc)
- [**W3 Schools: Python**](https://www.w3schools.com/python/)
- [**Another YT playlist**](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mFu00qxl0a67RhjjZj3jXm)

### Jupyter Notebooks

Instead of asking you to build traditional Python scripts, in this course you will be using **Jupyter Notebooks** to write and run your code.

Jupyter Notebooks allow you to divide your code into multiple cells that you can run individually one by one, allowing you to debug your code much more easily. You can go over the first few sections of [**this introductory document**](https://realpython.com/jupyter-notebook-introduction/) or [**this video**](https://www.youtube.com/watch?v=HW29067qVWk) (only until Notebook creation and code execution).

For this course, you will be using [**Google Colab**](https://colab.google/) to run your notebooks on the cloud. Ensure that you have a Gmail account to use it.

## NumPy Basics

NumPy (Numerical Python) is a linear algebra library in Python. It is a very important library that many popular libraries for AI/ML and data science (like Mat−plotlib, and SciPy) rely on.

What is it used for?
NumPy is very useful for performing mathematical and logical operations on Arrays. It provides an abundance of useful features for operations on n-arrays and matrices in Python.

One of the main advantages of Numpy is that vectorization using Numpy arrays makes it super time efficient. It enables parallel computation makes it so fast and hence extremely useful. Go through the following resources to familiarize yourself with them.

- [**Official Quickstart**](https://numpy.org/doc/stable/user/quickstart.html)
- **[W3 Schools: NumPy Tutorial](https://www.w3schools.com/python/numpy/default.asp) (Recommended)**
- [**FreeCodeCamp: Python NumPy Tutorial for Beginners**](https://www.youtube.com/watch?v=QUT1VHiLmmI&pp=ygUObnVtcHkgdHV0b3JpYWw%3D) (Video tutorial)

## Assignment

Python is an essential tool for this project as we will be putting our intuitive thoughts about natural language understanding into logical structures of rules that the computer will perform via Python language, this assignment is based on your proficiency in Python

### Problem Statement:

You are a given an integer array A of length n, calculate the sum of squares of all the non-negative elements of A.

**You are not allowed to use while loop, for loop, goto statements, and iterators; statements such as the ones given below are not allowed**
* [i*i for i in A]
* sum(A)

**Your program should take the input from a text file and print the output in a separate text file, examples of both of which are given below**

### Input

The first line of the input contains the number of test cases T, for each test case the first line contains the length of the array N and the second line contains the space-separated elements of the array

### Output

For each array print the sum of squares of all the non-negative elements of A

### Submission

Provide a link to your submission Python script or Jupyter Notebook in the submission form (The link will be provided soon)

## Neural Networks

### **Strongly Recommended**

**Finish [this course "Neural Networks and Deep Learning"](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning) by Andrew Ng on Coursera and complete the video tutorials by the end of this week.**

You do not need to enroll for the course, you can audit all the content for free.

- You may also watch [**this 4 video infographic series by 3Blue1Brown**](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) to better understand a lot of the math behind neural networks

## PyTorch

PyTorch is another powerful library we'll be using to implement our neural networks. Again, you do not require an in-depth knowledge of this module, just a basic understanding of the syntax and the specific functions we'll be using.

- [**PyTorch tutorial**](https://pytorch.org/tutorials/beginner/basics/intro.html) for the basic functions you'll need to build the classifier.
- [**Complete video tutorial**](https://www.youtube.com/watch?v=c36lUUr864M&pp=ygUcbmV1cmFsIG5ldHdvcmsgd2l0aCBweXRvcmNoIA%3D%3D)

## Basics of NLP

Here are some links to resources on this week's topics. Please go through  them.

- [Natural Language Processing (NLP)](<#natural-language-processing-(nlp)>)
  - [Word Embeddings](#word-embeddings)
  - [Text Preprocessing](#text-preprocessing)

## Natural Language Processing (NLP)

Natural Language Processing is the AI technique by which computers are trained to recognize, understand, and reply in human 'natural' languages. Refer to the following links to get a brief understanding of NLP and the concepts behind it.

- [**Short introductory video**](https://youtu.be/CMrHM8a3hqw)
- [**Advanced textbook by Dan Jurafsky**](https://web.stanford.edu/~jurafsky/slp3/), **OPTIONAL**, for those who are interested
- [**Youtube Playlist by codebasics**](https://www.youtube.com/playlist?list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX) (contains extra content)

### Word Embeddings

Neural networks cannot process words directly; they deal only with numerical vectors and their computations. To feed text as input to a neural network, we will first need to convert it into vector form, using word embeddings. There exist many different techniques (TF-IDF, Skip-gram, CBOW) and implementations (Glove, FastText, etc.) for this purpose.

- [**Brief conceptual overview**](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)
- [**Another conceptual site**](https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314), specifically for skip gram models.

### Text Preprocessing

Real-word text is often not in a format appropriate for analysis. Things like spelling mistakes, punctuations, tenses, and more complex concepts of a language are difficult to translate into vector representations. To clean up and simplify the input data, we use several preprocessing techniques. Refer to the below link on the same.

- [**Detailed guide**](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/) (Recommended)
- [**Youtube Video**](https://www.youtube.com/watch?v=nxhCyeRR75Q), focusing more on post-cleanup steps like tokenization, stemming, and lemmatization

### Graded Assignment

This week's graded assignment has been released. Open [NN_using_PyTorch_(Graded).ipynb](https://colab.research.google.com/drive/1cKa6JQiwuZlvBwWkFlZQo7S8gWYDZe98) in Colaboratory, and save a copy of the notebook to your Drive.

Download the pizza_vs_not.zip dataset, unzip and upload it to your Drive.

Make sure that you save both the Colab Notebook and dataset folder to the same account.
Submissions will be via Google Forms, the link will be shared later.

## Week-2

After understanding the basics of NLP, and how we can convert words into processable embeddings, we must now understand how we can begin to properly process sentences and then go forward whole bodies of text.  
Until now, while we have been including information gained from the number and positions of words in sentences, we have not been able to use the _order_ or, more precisely, the **_sequence_** of these words in a sentence.

Now, you will understand the basic concepts behind Recurrent Neural Networks and their working, and use them in this week's assignment to process text. This will be a relatively light week, theory-wise.

Here are the links to resources on this week's topics. Please go through all of them carefully. The graded assignment for this week has already been uploaded.

- [RNN - Coursera Course](#coursera-course)
- [Python Classes](#python-classes)
- [Optimizers](#optimizers)
- [Assignment](#assignment)

## Coursera Course

### **Strongly Recommended**

**Finish [the first two weeks of this course "Sequence Models"](https://www.coursera.org/learn/nlp-sequence-models?) by Andrew Ng on Coursera.**

It contains all the theory behind the concepts that will be covered this week. Week 1 of the course is on sequential neural networks, while Week 2 revisits word embeddings.

You do not need to enroll for the course, you can audit all the content for free.

## Python Classes

Python classes offer a means of bundling data and functionality together, providing you with a simple interface for you to use. Python is an object-oriented programming language.

Almost everything in Python is an object, with its properties and methods. Classes are like object constructors, a "blueprint" for creating objects. Here are some resources to give you a basic overview of classes.

- [**Short Introduction**](https://www.geeksforgeeks.org/python-classes-and-objects/)
- [**PyTorch Docs - Implementing models using classes**](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)

## Optimizers

Optimizers play a crucial role in training deep learning models. They are algorithms that adjust the parameters of neural networks during the learning process to minimize the error or loss function. The primary goal of optimizers is to find the optimal set of parameters that best fit the given training data and generalize well to unseen data, with as little computation as possible.

There are various optimization algorithms designed to converge to the optimal solution. A few of the popular optimizers used in deep learning include:

- SGD (Stochastic Gradient Descent):  
  This is the fundamental optimizer in deep learning. It updates the parameters after each training sample, making it computationally efficient. However, it may oscillate around the optimal solution due to its noisy updates.

- Adam (Adaptive Moment Estimation):  
  Adam uses adaptive learning rates for each parameter and maintains an exponentially decaying average of past squared gradients and past gradients. Adam is widely used and often performs well in practice.

Here are a few resources on the topic, although only an understanding of the trade-offs between techniques is required.

- [**PyTorch Docs - Optimizers**](https://pytorch.org/docs/stable/optim.html)
- [**Explanation of the most common optimizers**](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6)
- [**CS231n Material**](https://cs231n.github.io/neural-networks-3/#update)

## Assignment

This week's assignment has been released. Open [[**Sentiment_Analysis_using_LSTM_(Graded).ipynb**](https://colab.research.google.com/drive/1l1dxNj0NfBgdoSiIE-f8AB_syvqYa7sK?usp=sharing)] in Colaboratory, and save a copy of the notebook to your Drive.

You will also have to download the `reviews.csv` dataset from [**this link**](https://drive.google.com/file/d/1nqmfdx7dj5qgynVwzD1CMjFFZoBeKrmD/view?usp=drive_link) and upload it to your Drive.

**Make sure that you save both the Colab Notebook and dataset to the same account**

