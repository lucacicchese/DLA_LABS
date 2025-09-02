# DLA_LABS

## Contents

1. [Completed exercises](#completed-exercises)
2. [Detailed file structure](#file-structure)
3. [Environment](#environment)
4. [Exercise 1](#exercise-1---distilbert-on-rotten-tomatoes)
5. [Exercise 2](#exercise-2---fine-tuning-distilbert)
6. [Exercise 3](#exercise-32---fine-tuning-clip)

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
│   clip.py
│   LoRA.py
│   utils.py
│   

 ```

## Environment

The testing environment has been managed with anaconda:
`conda env create -f environment.yml`

## Exercise 1 - DistilBERT on rotten tomatoes

In this exercise I implemented a text classification pipeline for sentiment analysis using the DistilBERT model and a Support Vector Classifier (SVC) as the final classifier. The main objective is to extract features from the text data using a pre-trained transformer model (DistilBERT), train an SVC classifier on these features, and then evaluate its performance on the validation and test sets.

### DistilBERT

DistilBERT is a distilled version of BERT. This means that knowledge was transfered to this new smaller model retaining as much as possible the performance of the bigger one. DistilBERT has about 66M parameters compared to the 110M of the normal BERT but still performs almost as well.
BERT is a model mainly used for text classification, sentiment analysis and similar tasks.

### Rotten Tomatoes dataset

The dataset used in this exercise has been the Cornell Movie Reviews dataset. I also explored this dataset using the documentation provided on [huggingface](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/viewer) an I learnt that:

- it has about 8.53k examples in the training set
- it has about 1.07k examples in the validation and test sets
- each set is perfectly balanced with 50% positive and 50% negative reviews
- negative reviews have been labeled with 0 and positive ones with 1
- the minimum string lenth is 4 characters and  maximum length is 267

### Implementation 1

The script starts by defining a configuration dictionary that contains hyperparameters for the experiment, including the dataset name, model name, batch size, learning rate, and the logging configurations for TensorBoard and WandB.
Depending on the configuration, the script initializes logging to Weights and Biases (WandB) and TensorBoard. These logging frameworks track the experiment and visualize metrics during training and evaluation.
The dataset `rotten_tomatoes` is loaded using the Hugging Face datasets library and then split using the available splits into train, validation, and test sets.
The DistilBERT model is then loaded using the AutoModel and AutoTokenizer classes from the transformers library. A pipeline is created to perform feature extraction, which will extract embeddings for text inputs to use in the SVC model.
The script uses the function extract_features to extract features from the dataset using the pre-trained model.
After extracting features from the text data, an SVC classifier is trained using the extracted training features and labels.
Once the classifier is trained, the performance gets evaluated on both the validation and test sets. The evaluation metrics include accuracy and detailed classification reports.

### Results 1

After training the SVC classifier on the extracted features the performance was evaluated:

| Metric                    | Validation Set                 | Test Set                        |
|---------------------------|--------------------------------|---------------------------------|
| Accuracy                  | 0.82                           | 0.79                            |
| Precision                 | 0.82                           | 0.80                            |
| Recall                    | 0.81                           | 0.79                            |
| F1 Score                  | 0.81                           | 0.79                            |

## Exercise 2 - Fine-tuning Distilbert

In this exercise I implemented a sequence classification pipeline for sentiment analysis using the DistilBERT model using the Trainer class from Hugging Face's transformers library. The model is fine-tuned on the Rotten Tomatoes dataset for sentiment classification. The script also integrates Weights and Biases (WandB) and TensorBoard for logging experiment details, monitoring the training process, and visualizing performance metrics.

### Implementation 2

The script starts by defining a configuration dictionary that holds various hyperparameters and paths for the experiment.
The dataset is loaded using the load_dataset function from the Hugging Face datasets library.
A tokenizer is then loaded for DistilBERT, and the dataset is tokenized into a format suitable for training, including padding and truncation of text to the maximum sequence length specified.
The tokenized datasets include input IDs and attention masks for each text example. These are essential for feeding data into transformer models like DistilBERT.
A DistilBERT model for sequence classification is loaded using the AutoModelForSequenceClassification class. The number of classes in the dataset is determined from the training labels, which is then passed to the model as a parameter for the number of output labels.
The DataCollatorWithPadding is used to dynamically pad input sequences to the maximum length. This ensures that the model can efficiently process batches of varying lengths without requiring a fixed sequence length for all examples.
Training arguments are specified in the TrainingArguments class, where settings like learning rate, batch size, number of epochs, and logging frequency are defined. The arguments also specify how often to save the model and log training metrics. The most critical parameters can be set in the *config* dictionary in the main file to make it more convenient to change them if needed.
The script then initializes a Trainer object, which handles the training loop, evaluation, and logging automatically.
The metrics are computed with *compute_metrics* that calculates accuracy, precision, recall and F1 score using the sklearn functions.

### Results 2

The model was trained on the Rotten Tomatoes dataset using DistilBERT for feature extraction and fine-tuned for sentiment classification. The following results were obtained:

| Metric                    | Validation Set                 | Test Set                        |
|---------------------------|--------------------------------|---------------------------------|
| Accuracy                  | 0.85227                        | 0.84186                         |
| Precision                 | 0.87033                        | 0.87500                         |
| Recall                    | 0.83114                        | 0.80113                         |
| F1 Score                  | 0.85029                        | 0.83643                         |

This shows a significant improvement compared to the results of the previous exercise showing that the training was completed effectively.

## Exercise 3.2 - Fine-tuning CLIP

Fine tuning CLIP proved to be a much greater challenge than fine-tuning DistilBERT, in this exercise I dealt with different problems. When I initially tried to use the map function on the dataset it filled all the available disk space causing the program to crash. This was resolved by switching to lazy mapping, processing the data only when needed, which helped manage memory usage more effectively. Further into the process, I faced additional memory constraints while training the model and concatenating the logits. The issue arose due to system memory limitations, particularly when processing large batches. To address this, I used a custom data collator to ensure efficient handling of the input data during batching. This also involved reducing the batch size during training, which helped alleviate memory stress.

### CLIP

CLIP is a is a multimodal vision and language model that aligns in the same embedding space both image and text encodings. It can be used for zero-shot image classification, generating images descriptions or finding images based on a given description.

### TinyImagenet dataset

The dataset used for training and evaluation is the TinyImageNet dataset, which was split into training and validation sets. Each image in the dataset is paired with a label that corresponds to a class, and we aim to train a model that can predict the correct class for each image given its corresponding text prompt `“a photo of a [class_name]”`.

To efficiently handle the dataset and avoid memory overload, I implemented a custom LazyDataset class. This class lazily loads data when necessary, using lazy mapping instead of processing the entire dataset upfront, this helps with memory management. The `__getitem__` method processes images and corresponding labels on demand, using a text prompt like `"a photo of a [class_name]"` for each image label, and feeds this data into the CLIP processor.

### Implementation 3

First of all to establish a baseline I used CLIP on the dataset to test it's zero-shot capabilities. Then I moved on to fine-tuning CLIP to get the most out of it.

To be able to run this fine-tuning a lot of custom solutions proved necessary.

To fine-tune the model in a parameter efficent way, I applied LoRA (Low-Rank Adaptation), which focuses on fine-tuning specific parts of the model (like the query and value projection layers) rather than training the entire model. This significantly reduces the number of parameters being updated and helps prevent memory exhaustion during training.
The LoRA configuration specifies which parts of the model to fine-tune, in this case, the query projection (`q_proj`) and value projection (`v_proj`) layers. This was done using the LoRAConfig and `get_peft_model` function from the `peft` library. I chose these parts because in the original LoRA paper the authors established that injecting into the attention-related weight matrices is highly effective and it doesn't require too much compute power.

I also had to use a custom collator and trainer. In the training process, the model expects a batch of data consisting of both images and their corresponding text inputs. I implemented a custom collator function `clip_collator` to ensure that the data is batched efficiently.

I had to write a custom `CLIPTrainer` with a `compute_loss` function that calculates the contrastive loss by comparing the similarity between the image and text embeddings by taking the diagonal of the similarity matrix and useing it to compute a cross-entropy loss for image-text pairs.

The training process was managed using Hugging Face’s Trainer API. The custom CLIPTrainer was created to accommodate the contrastive loss and other custom behavior, such as caching text features for each class.

Once training was complete, I evaluated the model's performance using several metrics, including accuracy, precision, recall, and F1 score.

For tracking the experiment, I used both WandB and TensorBoard to log metrics, training progress, and visualizations.

### Results 3

The fine-tuning of the CLIP model using LoRA (Low-Rank Adaptation) demonstrated a significant improvement over the zero-shot baseline.

Before any fine-tuning, I evaluated the model on the validation set using a zero-shot approach, where the model made predictions based on pre-trained features without any further training. The results for the zero-shot baseline were as follows:

- Accuracy: 0.6425
- Precision: 0.7037
- Recall: 0.6425
- F1 Score: 0.6417

This baseline provided a solid starting point for comparison, showing that the pre-trained CLIP model was already performing reasonably well (as expected) on the TinyImageNet dataset.

After fine-tuning the model using LoRA, which involved training just 491,520 parameters out of a total of 150,112,257 parameters. That means that by training approximately 0.33% of the parameters the model's performance showed considerable improvement:

- Accuracy: 0.7667
- Precision: 0.7747
- Recall: 0.7667
- F1 Score: 0.7669

The LoRA-based fine-tuning led to a 12.42% improvement in accuracy and a 12.15% increase in F1 score compared to the zero-shot baseline, demonstrating that the fine-tuning process was successful in improving the model's ability to classify images.

## Main take-aways

Throughout these exercises, I gained hands-on experience with fine-tuning transformer models, specifically DistilBERT for text classification and CLIP for image-text tasks. I learned how to use pre-trained models for feature extraction, fine-tune them for specific tasks, and efficiently manage resources, especially when dealing with memory constraints during model training.

I also explored how to handle challenges like contrastive learning for multimodal tasks with CLIP and optimized memory usage through techniques like lazy mapping and LoRA. Overall, these exercises provided me with practical skills in model fine-tuning and efficient training transformers.
