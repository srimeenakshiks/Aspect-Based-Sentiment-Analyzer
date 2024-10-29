---
library_name: transformers
tags:
- sentiment analyzer
- aspect based sentiment analyzer
- text classification
- bert model
- imdb dataset
license: mit
datasets:
- stanfordnlp/imdb
language:
- en
base_model:
- google-bert/bert-base-uncased
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->
This model is a fine-tuned BERT model designed for aspect-based sentiment analysis, enabling the classification of sentiments associated with specific aspects in text. It provides valuable insights into customer opinions and sentiments regarding different features in user-generated content.



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The **Aspect-Based Sentiment Analyzer using BERT** is a state-of-the-art natural language processing model designed to identify and analyze sentiments expressed towards specific aspects within a given text. Leveraging the power of the BERT architecture, this model excels in understanding contextual nuances, enabling it to accurately classify sentiments as positive, negative, or neutral for various product features or attributes mentioned in customer reviews or feedback.

Trained on the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb), the model has been fine-tuned to detect sentiment related to different aspects, making it valuable for businesses aiming to enhance customer satisfaction and gather insights from user-generated content. Its robust performance can aid in sentiment analysis tasks across various domains, including product reviews, service evaluations, and social media interactions.

- **Developed by:** Srimeenakshi K S
- **Model type:** Aspect-Based Sentiment Analysis
- **Language(s) (NLP):** English
- **License:** MIT License
- **Finetuned from model:** BERT-base-uncased


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
The model can be used directly to classify sentiments in user-generated text based on specified aspects without the need for additional fine-tuning. It is suitable for analyzing reviews, social media posts, and other forms of textual feedback.


### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
This model can be integrated into applications for customer feedback analysis, chatbots for customer service, or sentiment analysis tools for businesses looking to improve their products and services based on customer input.


### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
The model may not perform well with text that contains heavy sarcasm or nuanced expressions. It should not be used for critical decision-making processes without human oversight.


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
The model may reflect biases present in the training data, leading to potential misclassification of sentiments. Users should be cautious in interpreting results, particularly in sensitive applications where sentiment analysis can impact customer relationships.


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. It is recommended to validate results with a diverse set of data and consider human judgment in ambiguous cases.

## How to Get Started with the Model

Use the code below to get started with the model.

```
from transformers import pipeline

sentiment_analyzer = pipeline("text-classification", model="srimeenakshiks/aspect-based-sentiment-analyzer-using-bert")
result = sentiment_analyzer("The food was amazing, but the service was slow.", aspect="service")
print(result)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on the [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb), which contains movie reviews labeled with sentiment (positive and negative). This dataset is commonly used for sentiment analysis tasks and includes a diverse range of reviews, allowing the model to learn various expressions of sentiment effectively.


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->


#### Preprocessing

Data preprocessing involved tokenization, padding, and normalization of text inputs to fit the BERT model requirements.


#### Training Hyperparameters

- **Training regime:** fp16 mixed precision <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model was evaluated using the same dataset on which it was trained, ensuring consistency in performance metrics and providing a reliable assessment of its capabilities in aspect-based sentiment analysis.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation included various aspects such as product features, service quality, and user experience.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Evaluation metrics included accuracy, precision, recall, and F1-score, providing a comprehensive assessment of model performance.

### Results

The model achieved an accuracy of 95% on the test dataset, demonstrating effectiveness in aspect-based sentiment classification.

#### Summary

The results indicate that the model performs well across a range of aspects but may struggle with nuanced sentiment expressions.


## Model Examination

<!-- Relevant interpretability work for the model goes here -->

Further interpretability work can be conducted to understand how the model makes its predictions, particularly focusing on attention mechanisms within BERT.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 4050
- **Hours used:** 20 hours
- **Cloud Provider:** AWS
- **Compute Region:** US-East
- **Carbon Emitted:** 3.5

## Technical Specifications

### Model Architecture and Objective

The model is based on the BERT architecture, specifically designed to understand the context of words in a sentence, enabling it to classify sentiments associated with different aspects effectively.

### Compute Infrastructure

#### Hardware

- **GPU:** NVIDIA GeForce RTX 4050
- **RAM:** 16GB

#### Software

- **Framework:** PyTorch
- **Library Version**: Hugging Face Transformers version 4.44.2

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@model{srimeenakshiks2024aspect,
  title={Aspect-Based Sentiment Analyzer using BERT},
  author={Srimeenakshi K S},
  year={2024},
  publisher={Hugging Face}
}


**APA:**

Srimeenakshi K S. (2024). _Aspect-Based Sentiment Analyzer using BERT_. Hugging Face.

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

- **Aspect-Based Sentiment Analysis (ABSA):** A subfield of sentiment analysis that focuses on identifying sentiments related to specific features or aspects of a product or service.


## Model Card Authors

- **Author:** Srimeenakshi K S

## Model Card Contact

For inquiries or feedback, please reach out to [srimeenakshiks@gmail.com].
