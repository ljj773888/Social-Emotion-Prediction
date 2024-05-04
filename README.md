# Social-Emotion-Prediction
A collection of papers on the field of **Social Emotion Prediction**.

Social emotion differs from the emotion expressed by the text's writer, as it reflects the collective emotional response of the audience to the content. The task involves identifying which emotions a text is likely to provoke in its readers, considering that these responses can be influenced by individual backgrounds and personal experiences, which are often unknown.

## Problem Definition

### Given Data:
- **Articles**: A set of news articles $\( D = \{d_1, d_2, \ldots, d_{|D|}\} \)$.
- **Comments**: Corresponding sets of comments $\( C = \{c_1, c_2, \ldots, c_{|D|}\} \)$, each comment set $\( c_i \)$ is associated with an article $\( d_i \)$.

### Emotion Tags:
- A set of predefined emotion tags $\( E = \{e_1, e_2, \ldots, e_{|E|}\} \)$ used for emotional classification.

### Social Emotion Distribution:
- Each article is associated with an emotion rating distribution across these tags, normalized and represented as $\( l_i = [l_{i,e1}, l_{i,e2}, \ldots, l_{i,e|E|}] \)$.

### Article Representation:
- Each article $\( d_i \)$ is divided into blocks $\( \{b_1, b_2, \ldots, b_{|d_i|}\} \)$, each block containing words $\( \{w_1, w_2, \ldots, w_{|b_j|}\} \)$.

### Comment Representation:
- The comments for an article $\( c_i \)$ are represented as a sequence of words from all comments combined.

The goal is to predict the social emotion distribution $\( l_i \)$ for each article based on its content and associated comments.

## Dataset & Surveys
- (*NLPCC'20*) A Large-Scale Chinese Short-Text Conversation Dataset [[paper](https://arxiv.org/abs/2008.03946)] [[code](https://github.com/huggingface/datasets)]
- (*ACL'20*) Hashtags, Emotions, and Comments: A Large-Scale Dataset to Understand Fine-Grained Social Emotions to Online Topics [[paper](https://aclanthology.org/2020.emnlp-main.106/)] [[code](https://github.com/polyusmart/HEC-Dataset)]
- (*ACL'20*) Goodnewseveryone: A corpus of news headlines annotated with emotions, semantic roles, and reader perception. [[paper](https://aclanthology.org/2020.lrec-1.194/)] 
- (*IEEE Transactions on Computational Social Systems*) LCSEP: A Large-Scale Chinese Dataset for Social Emotion Prediction to Online Trending Topics [[paper](https://ieeexplore.ieee.org/document/10379492)]
- (*DATA'21*) A Survey of Social Emotion Prediction Method [[paper](https://researchr.org/publication/AlsaediBGT21)]



