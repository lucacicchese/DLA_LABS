# DLA_LABS

## Contents

1. [Completed exercises](#completed-exercises)
2. [Detailed file structure](#file-structure)
3. [Environment](#environment)
4. [Exercise 1](#exercise-1---sentiment-analysis)
5. [Exercise 2](#exercise-2---fine-tuning-distilbert)
6. [Exercise 3](#exercise-32---fine-tuning-a-clip-model)

## Completed exercises

|  Exercise   | DONE  | WIP |
|-----|---|---|
| LAB03 Exercise 1 | ✅ | |
| LAB03 Exercise 2 | ✅ | |
| LAB03 Exercise 3 | ✅ | |

## File Structure

```linux
LAB03
│   README.md
│   environment.yml
│   exercise_1.py
│   exercise_2.py
│   exercise_3.py
│   evaluate.py
│   models.py
│   
└───logs
    └─── checkpoints
    └─── tensorboard
 ```

## Environment

The testing environment has been managed with anaconda:
`conda env create -f environment.yml`

## Exercise 1 - DistilBERT on rotten tomatoes

In this exercise I performed sentiment analysis on the Cornell Rotten Tomatoes movie review dataset. I incrementally built the sentiment analysis pipeline using DistilBERT as a feature extractor and then training an SVM used as a classifier.

### Implementation 1

I initially loaded the dataset and verified the available splits as required. I also explored the dataset further using the documentation provided on [huggingface](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/viewer) learning that:

- it has about 8.53k examples in the training set
- it has about 1.07k examples in the validation and test sets
- each set is perfectly balanced with 50% positive and 50% negative reviews
- negative reviews have been labeled with 0 and positive ones with 1
- the minimum string lenth is 4 characters and  maximum length is 267
I then proceded to load the pre-trained `distilbert-base-uncased` model and corresponding tokenizer.

### Results 1

## Exercise 2 - Fine-tuning Distilbert

### Implementation 2

### Results 2

## Exercise 3.2 - Fine-tuning CLIP

### Implementation 3

### Results 3
