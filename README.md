# Social-Emotion-Prediction
A collection of papers on the field of "Social Emotion Prediction".

This field is an extension of sentiment analysis, going beyond simple positive, negative, or neutral classifications to understand deeper emotional impacts of texts on readers, which focuses on the task of predicting the emotions of readers when they are exposed to a text, referred to as social emotion prediction. This field is an extension of sentiment analysis, going beyond simple positive, negative, or neutral classifications to understand deeper emotional impacts of texts on readers.

Social emotion differs from the emotion expressed by the text's writer, as it reflects the collective emotional response of the audience to the content. The task involves identifying which emotions a text is likely to provoke in its readers, considering that these responses can be influenced by individual backgrounds and personal experiences, which are often unknown. Specifically:

### Problem Definition

### Problem Definition

#### Given Data
- **Articles**: A set of news articles \( D = \{d_1, d_2, \ldots, d_{|D|}\} \).
- **Comments**: Corresponding sets of comments \( C = \{c_1, c_2, \ldots, c_{|D|}\} \), each comment set \( c_i \) is associated with an article \( d_i \).

#### Emotion Tags
- A set of predefined emotion tags \( E = \{e_1, e_2, \ldots, e_{|E|}\} \) used for emotional classification.

#### Social Emotion Distribution
- Each article is associated with an emotion rating distribution across these tags, normalized and represented as:
  ![li = [l_{i,e1}, l_{i,e2}, \ldots, l_{i,e|E|}]](https://latex.codecogs.com/svg.latex?l_i%20=%20[l_{i,e1},%20l_{i,e2},%20\ldots,%20l_{i,e|E|}])

#### Article Representation
- Each article \( d_i \) is divided into blocks \( \{b_1, b_2, \ldots, b_{|d_i|}\} \), each block containing words \( \{w_1, w_2, \ldots, w_{|b_j|}\} \).

#### Comment Representation
- The comments for an article \( c_i \) are represented as a sequence of words from all comments combined.

The goal is to predict the social emotion distribution \( l_i \) for each article based on its content and associated comments.





