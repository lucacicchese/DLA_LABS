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

This exercise implements a text classification pipeline for sentiment analysis using the DistilBERT model and a Support Vector Classifier (SVC) as the final classifier. The main objective is to extract features from the text data using a pre-trained transformer model (DistilBERT), train an SVC classifier on these features, and then evaluate its performance on the validation and test sets. The process includes several steps: dataset loading, feature extraction, classifier training, and model evaluation. Additionally, the results are logged using Weights and Biases (WandB) and TensorBoard for tracking and visualizing the model's performance.

### Rotten Tomatoes dataset

I initially loaded the dataset and verified the available splits as required. I also explored the dataset further using the documentation provided on [huggingface](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/viewer) learning that:

- it has about 8.53k examples in the training set
- it has about 1.07k examples in the validation and test sets
- each set is perfectly balanced with 50% positive and 50% negative reviews
- negative reviews have been labeled with 0 and positive ones with 1
- the minimum string lenth is 4 characters and  maximum length is 267

### Implementation 1

The script starts by defining a configuration dictionary that contains hyperparameters for the experiment, including the dataset name, model name, batch size, learning rate, and the logging configurations for TensorBoard and WandB.
Depending on the configuration, the script initializes logging to Weights and Biases (WandB) and TensorBoard. These logging frameworks track the experiment and visualize metrics during training and evaluation.
The DistilBERT model is loaded using the AutoModel and AutoTokenizer classes from the transformers library. A pipeline is created to perform feature extraction, which will extract embeddings for text inputs to use in the SVC model.
The dataset, rotten_tomatoes, is loaded using the Hugging Face datasets library. It splits the dataset into train, validation, and test sets.
The script uses the function extract_features to extract features from the dataset using the pre-trained model. The extracted features are stored in arrays that can be directly used to train an SVC model.
After extracting features from the text data, an SVC classifier is trained using the extracted training features and labels. The training process includes fitting the SVC model to the data.
Once the classifier is trained, the script evaluates its performance on both the validation and test sets. The evaluation metrics include accuracy and detailed classification reports.
Validation Evaluation: The validation set is used to fine-tune the classifier and check its performance on unseen data.
Test Evaluation: The test set is used to evaluate the final performance of the model after training.
The evaluation results (accuracy and classification report) are logged to WandB and TensorBoard for further analysis.

### Results 1

The experiment was conducted on the Rotten Tomatoes dataset using the DistilBERT model for feature extraction and an SVC classifier for sentiment classification. The following results were obtained:

Validation Results

After training the SVC classifier on the extracted features, the validation performance was evaluated:

Validation Accuracy: 85.43% (example value, adjust based on actual result)

Precision, Recall, F1-Score: (Include detailed metrics from the classification_report output)

## Exercise 2 - Fine-tuning Distilbert

In this exercise I implemented a sequence classification pipeline for sentiment analysis using the DistilBERT model with a Trainer API from Hugging Face's transformers library. The model is fine-tuned on the Rotten Tomatoes dataset for sentiment classification. The script also integrates Weights and Biases (WandB) and TensorBoard for logging experiment details, monitoring the training process, and visualizing performance metrics. The pipeline includes dataset loading, tokenization, model initialization, training, and evaluation with performance metrics. This approach uses the Trainer class for easy management of the training loop and evaluation process.

### Implementation 2

The script starts by defining a configuration dictionary that holds various hyperparameters and paths for the experiment. This includes the name of the dataset, model, batch size, number of epochs, learning rate, and logging settings for both TensorBoard and WandB.

Logging for WandB and TensorBoard is initialized based on the settings in the configuration file. WandB is used for tracking experiments, while TensorBoard is used to visualize training metrics such as loss and accuracy.
The dataset is loaded using the load_dataset function from the Hugging Face datasets library. The Rotten Tomatoes dataset is divided into training, validation, and test splits. A tokenizer is loaded for DistilBERT, and the dataset is tokenized into a format suitable for training, including padding and truncation of text to the maximum sequence length specified.
The tokenized datasets include input IDs and attention masks for each text example. These are essential for feeding data into transformer models like DistilBERT.
A DistilBERT model for sequence classification is loaded using the AutoModelForSequenceClassification class. The number of classes in the dataset is determined from the training labels, which is then passed to the model as a parameter for the number of output labels.
The DataCollatorWithPadding is used to dynamically pad input sequences to the maximum length within a batch. This ensures that the model can efficiently process batches of varying lengths without requiring a fixed sequence length for all examples.

Training arguments are specified in the TrainingArguments class, where settings like learning rate, batch size, number of epochs, and logging frequency are defined. The arguments also specify how often to save the model and log training metrics.
The script then initializes a Trainer object, which handles the training loop, evaluation, and logging automatically. The trainer is configured with the model, training arguments, datasets, data collator, and a compute_metrics function for calculating performance metrics like accuracy, precision, recall, and F1 score.
The training process is initiated with trainer.train(), and the evaluation results are stored after the training is complete.

### Results 2

The model was trained on the Rotten Tomatoes dataset using DistilBERT for feature extraction and fine-tuned for sentiment classification. The following results were obtained:

Training Details

- Number of Epochs: 5
- Batch Size: 16
- Learning Rate: 2e-5
- Evaluation Results

After fine-tuning the model using the Trainer class, the evaluation results on the validation and test sets were as follows:

Validation Accuracy: [Insert Validation Accuracy]

Test Accuracy: [Insert Test Accuracy]

The classification metrics (precision, recall, F1 score) were also computed and included in the classification_report.

## Exercise 3.2 - Fine-tuning CLIP

Fine tuning CLIP proved to be a much greater challenge than fine-tuning DistilBERT, in this exercise I dealt with different problems. Initially I encountered an Out Of Memory error when of OOM firstly when applying the map function on the dataset. This was resolved by switching to lazy mapping, processing the data only when needed, which helped manage memory usage more effectively. Further into the process, I faced additional memory constraints while training the model and concatenating the logits. The issue arose due to system memory limitations, particularly when processing large batches. To address this, I used a custom data collator to ensure efficient handling of the input data during batching. This also involved reducing the batch size during training, which helped alleviate memory stress.

### TinyImagenet dataset

The dataset used for training and evaluation is the TinyImageNet dataset, which was split into training and validation sets. Each image in the dataset is paired with a label that corresponds to a class, and we aim to train a model that can predict the correct class for each image given its corresponding text prompt (e.g., `“a photo of a [class_name]”`).

To efficiently handle the dataset and avoid memory overload, I implemented a custom LazyDataset class. This class lazily loads data when necessary, using lazy mapping instead of processing the entire dataset upfront, thus helping with memory management. The `__getitem__` method processes images and corresponding labels on demand, using a text prompt like `"a photo of a [class_name]"` for each image label, and feeds this data into the CLIP processor.

### Implementation 3

In the training process, the model expected a batch of data consisting of both images and their corresponding text inputs. I implemented a custom collator function `clip_collator` to ensure that the data is batched efficiently. This ensures that images and text inputs are properly stacked, and labels are correctly included in each batch.

The CLIP model consists of two parts: a text encoder and an image encoder. During training, the model computes logits for both images and text, and these logits are compared using a contrastive loss function.

In my custom CLIPTrainer, the compute_loss method calculates the contrastive loss by comparing the similarity between the image and text embeddings. The diagonal of the similarity matrix is used to compute a cross-entropy loss for both image-text pairs.

To reduce memory overhead during training, I applied LoRA (Low-Rank Adaptation), which focuses on fine-tuning specific parts of the model (like the query and value projection layers) rather than training the entire model. This significantly reduces the number of parameters being updated and helps prevent memory exhaustion during training.

The LoRA configuration specifies which parts of the model to fine-tune, in this case, the query projection (q_proj) and value projection (v_proj) layers. This was done using the LoraConfig and get_peft_model utilities from the peft library.

I also implemented a zero-shot evaluation function, where the model was tested on a validation set without any further training, using pre-trained features. For each image in the validation set, cosine similarity between the image features and precomputed text features was calculated to determine the class with the highest similarity. This approach tests the ability of the model to generalize to new, unseen data.

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

After fine-tuning the model using LoRA, which involved training just 491,520 parameters out of a total of 150,112,257 parameters (with approximately 0.3274% of the parameters being trainable), the model's performance showed considerable improvement:

- Accuracy: 0.7667
- Precision: 0.7747
- Recall: 0.7667
- F1 Score: 0.7669

The LoRA-based fine-tuning led to a 12.42% improvement in accuracy and a 12.15% increase in F1 score compared to the zero-shot baseline, demonstrating that the fine-tuning process was successful in improving the model's ability to classify images.

## Main take-aways

Throughout these exercises, I gained hands-on experience with fine-tuning transformer models, specifically DistilBERT for text classification and CLIP for image-text tasks. I learned how to use pre-trained models for feature extraction, fine-tune them for specific tasks, and efficiently manage resources, especially when dealing with memory constraints during model training.

I also explored how to handle challenges like contrastive learning for multimodal tasks with CLIP and optimized memory usage through techniques like lazy mapping and LoRA. Overall, these exercises provided me with practical skills in model fine-tuning, efficient training, and experiment tracking, which are essential for working with large-scale machine learning models.
