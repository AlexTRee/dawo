---
language: code
license: mit
library_name: pytorch
tags:
- variational-autoencoder
- drug-response
- vae
- cancer-drug
- tahoe-deepdive
datasets:
- biomedical
- tahoebio/Tahoe-100M
---

# Ranking
### 2nd Place Award ($10,000 AWS credits)


# DAWO: Drug-Aware and Cell-line-Aware Variational Autoencoder


[![tahoe-deepdive](https://img.shields.io/badge/tag-tahoe--deepdive-blue)](https://huggingface.co/datasets/tahoebio/Tahoe-100M)


## Team Name
DAWO


## Members
- Yuhan Hao
- Sheng-Yong Niu
- Jaanak Prashar
- Tiange (Alex) Cui
- Danila Bredikhin
- Mikaela Koutrouli


## Project


### Title
DAWO: Drug-Aware and Cell-line-Aware Variational Autoencoder for Drug Response Prediction


### Overview
DAWO is a specialized Variational Autoencoder (VAE) designed to predict drug responses in cancer cell lines by integrating gene expression data with drug and cell line features. The model leverages multi-modal representation learning to capture complex interactions between drugs and cells, enabling more accurate prediction of drug responses across diverse conditions.


### Motivation
Understanding and predicting how cancer cells respond to different therapeutic compounds is crucial for advancing precision medicine approaches in oncology. Traditional methods often fail to capture the complex relationships between drugs, cell lines, and their molecular profiles. DAWO addresses this challenge by combining a VAE architecture with drug-aware and cell-line-aware components to model these interactions effectively.


### Methods
DAWO incorporates a multi-modal architecture with the following key components:


1. **Gene Expression Encoder**: Processes normalized gene expression data from cancer cell lines (input dimension: 5000)
2. **Drug Feature Encoder**: Processes drug features combining:
  - Drug summary embeddings
  - ChemBERTa molecular structure embeddings
  - Semantic feature embeddings
  (Total input dimension: 3122)
3. **Cell Line Feature Encoder**: Processes cell line features focusing on driver gene mutations and other genomic characteristics (input dimension: 113)
4. **Latent Space**: A 50-dimensional latent representation combining drug, cell line, and gene expression information
5. **Decoder**: Reconstructs gene expression profiles from the latent representation
6. **Classifier**: Predicts drug response categories from the latent representation (379 classes)


The model was trained using a combined loss function that balances reconstruction accuracy, latent space regularization, and classification performance.


### LLM Verification
There is also a LLM-as-judge component as the last step in the workflow before the results are evaluated by human experts. The LLM judge is [TxGemma](https://arxiv.org/pdf/2504.06196), a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability.
 

### Results
DAWO demonstrates strong performance in predicting drug responses across multiple cancer cell lines, with particular strength in:


1. Distinguishing between responsive and non-responsive cell lines for specific drugs
2. Generalizing to new drug-cell line combinations not seen during training
3. Capturing meaningful biological signals in the latent space that reflect known drug mechanisms and cellular pathways


### Discussion
Our model provides a powerful framework for drug response prediction that could accelerate drug discovery and repurposing efforts. The integration of multi-modal data (gene expression, drug features, cell line characteristics) enables DAWO to capture complex interaction patterns that simpler models miss.


Limitations include the need for comprehensive feature sets for new drugs and cell lines, and potential biases from the training data distribution. Future work will focus on incorporating additional molecular modalities and expanding the training data to improve generalization across diverse drug classes.


## Model Description
Using a variational autoencoder (VAE) approach, DAWO learns latent representations of these data sources and combines them to predict drug responses and identify potential drug-cell line interactions.


## Model Inputs and Outputs


### Inputs:
- **Gene Expression Data**: Normalized gene expression profiles (shape: [batch_size, 5000])
- **Drug Features**: Combined drug embeddings including:
 - Drug summary embeddings
 - ChemBERTa molecular structure embeddings
 - Semantic feature embeddings
 (Total shape: [batch_size, 3122])
- **Cell Line Features**: Cell line genomic profiles (shape: [batch_size, 113])


### Outputs:
- **Reconstructed Gene Expression**: Reconstructed expression profiles (shape: [batch_size, 5000])
- **Latent Representation**: Compressed representation in latent space (shape: [batch_size, 50])
- **Drug Response Predictions**: Predicted response classes (shape: [batch_size, 379])
- **Response Probabilities**: Softmax probabilities for each response class (shape: [batch_size, 379])


## How to Use


```python
from dawo_wrapper import DAWOWrapper


# Initialize model
model = DAWOWrapper(repo_path="path/to/model")


# Prepare inputs
# gene_expression: tensor of shape [batch_size, 5000]
# drug_features: tensor of shape [batch_size, 3122]
# cell_features: tensor of shape [batch_size, 113]


# Make predictions
results = model.predict(gene_expression, drug_features, cell_features)


# Access outputs
reconstructed_expression = results["x_hat"]
latent_representation = results["mu"]
drug_response_predictions = results["y_pred"]
response_probabilities = results["probs"]
```


## Download model from [Hugginface](https://huggingface.co/simonniu/dawo)
```
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/simonniu/dawo

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/simonniu/dawo
```


## Dataset
This model was developed using the [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) dataset as part of the Tahoe-DeepDive Hackathon 2025.


## License
MIT License

Copyright (c) 2025 Team DAWO

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

