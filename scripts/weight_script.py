import spacy
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load SpaCy model for POS tagging and dependency parsing
nlp = spacy.load('en_core_web_sm')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get weighted embedding
def get_weighted_embedding(sentence, weight_verb=0.5, weight_obj=0.5):
    doc = nlp(sentence)
    verb_weights = []
    obj_weights = []
    tokens = []
    
    # Extract tokens and assign weights based on POS and dependency
    for token in doc:
        if token.pos_ == 'VERB':
            verb_weights.append(weight_verb)
            tokens.append(token.text)
        elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj'):
            obj_weights.append(weight_obj)
            tokens.append(token.text)

    if not tokens:  # If no verbs or objects, fall back to sentence embedding without weights
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return sentence_embedding[0]
    
    # Tokenize and generate embeddings for each token
    token_embeddings = []
    for token in tokens:
        inputs = tokenizer(token, return_tensors='pt')
        outputs = model(**inputs)
        token_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        token_embeddings.append(token_embedding[0])
    
    # Combine weights
    weights = np.array(verb_weights + obj_weights)
    
    # Apply weights to embeddings
    token_embeddings = np.array(token_embeddings)
    weighted_embeddings = token_embeddings * weights[:, np.newaxis]
    sentence_embedding = np.mean(weighted_embeddings, axis=0)
    return sentence_embedding

# Define your sentences
sentences = ["send text", "send high definition picture", "receive text"]

# Generate weighted embeddings for the sentences
weighted_embeddings = [get_weighted_embedding(sentence) for sentence in sentences]

# Perform hierarchical clustering
Z = linkage(weighted_embeddings, 'ward')

# Plot the dendrogram
plt.figure()
dendrogram(Z, labels=sentences)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sentences')
plt.ylabel('Distance')
plt.show()
