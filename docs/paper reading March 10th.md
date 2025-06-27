## **《Attention is all you need》**

**One of the reasons that bias is embodied in transformer-based models:**

In transformer-based models, the self-attention layer is responsible for weighing the importance of each token in a sequence relative to every other token. This weighing can capture context and dependencies across the entire input. However, if there is bias in the training data, the model would be trained or fine-tuned to capture dependencies in a biased way. For example, if the training material consists of conversations from the middle of the last century, in a sentence containing "man," "woman," and "CEO," the dependencies between "CEO" and "man" would likely be higher than the dependencies between "woman" and "CEO." Some papers have proven that we can effectively debias by making adjustments in the attention mechanism of the model. [^1]

[^1]:Debiasing Attention Mechanism in Transformer without Demographics [link](https://openreview.net/forum?id=jLIUfrAcMQ)

## **《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》**

**What is the difference between Bert and transformer from an embedding bias perspective？**

BERT is a model that trains only the encoder of a Transformer architecture using a special training method. The "special training method" here is train the model in two task:

1. Randomly mask a word in a context and predict the masked word.
2. given a pair of sentence, predict whether the second sentence logically follows the first one in the original text.

The reason that we can train the model on these tasks where we do prediction based on the whole content is that when throw the decoder of transformer away, we also throw the constraint of only seeing words before the newest word when training the model away. This allows BERT to learn stronger associations between words in full context, potentially amplifying co-occurrence biases.

## 《RoBERTa: A Robustly Optimized BERT Pretraining Approach》

1. The aspects in RoBERTa's training methodology that might mitigate bias:
   1. The diversity of data and amount of exposure to data
   2. The number of epochs that the model is exposed to each sample
   3. Dynamic masking
2. Need for dedicated fairness benchmarks and research on fairness analysis in major NLP benchmarks:
   1. When reviewing the paper, it becomes apparent that quantitatively measuring fairness differences is challenging using the provided benchmarks, as these benchmarks have neither been proven fair nor tested specifically for fairness measurement.
   2. However, it can be reasonably deduced that RoBERTa did not exacerbate bias compared to BERT, given its superior performance across diverse benchmarks. It is unlikely that all these benchmarks would suffer from the same bias.

### 《Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings》

**Try to describe debias algorithm in my own words (except for the Soft bias correction part)**

1. We manually categorize some words in our vocabulary and use these categorized word sets to compute the covariance matrix of these categories.  
2. Perform Singular Value Decomposition (SVD) on this covariance matrix to obtain $k$ vectors, which represent the top $ k $ most prominent bias directions.  
3. We use these vectors to construct a new subspace $ B $ and perform debiasing operations using  $ B $. The debiasing operations mitigate bias in word embeddings along the  $ B $-direction.  
4. The specific debiasing methods are neutralize and equalize:  
   - Neutralize modifies the embedding vector of a word by projecting it onto the subspace orthogonal to  $ B $. This ensures the word no longer retains information along $ B $, thereby removing bias.  
   - Equalize adjusts the projections of paired words with opposing meanings (e.g., $A $ and $-A$) onto the subspace orthogonal to  $ B $. This ensures $A $ and $-A$ are equidistant from a neutral vector  $H $ (orthogonal to $ B $), eliminating bias between  $A $ and  $-A $ relative to  $H $.

### 《StereoSet: Measuring stereotypical bias in pretrained language models》

(I was surprised that the content of this paper corresponded to the comments I wrote for RoBERTa)

**Try to describe the evaluation metrics in my own words **:

In this article, a good model is defined as:  
1. The generated content is meaningful.  
2. The generated content is free of bias.  
Therefore, this article introduces the Idealized CAT Score (icat) metric, which is the product of the Language Modeling Score (lms) and the model’s neutrality(A terminology defined by me). A higher $ \text{lms} $ indicates more meaningful generated content.  
The $text{lms} $is defined as the probability that the model selects a meaningful continuation $ B_1$over an unmeaningful one $ B_2 $, given a context $A$ Formally:  
$$ \text{lms} = P(\text{model chooses } B_1 \mid A, B_1 \text{ (meaningful)}, B_2 \text{ (unmeaningful)}) $$ 
The model’s neutrality is defined as:  
$$ \frac{\min(\text{Stereotype Score (ss)}, 100 - \text{ss})}{50} $$  
The  $\text{ss}$  is defined similarly to $\text{lms}$, but replaces $B_1 $ and $ B_2 $ with stereotypical and anti-stereotypical sentences. When $\text{ss} = 50 $, the model has no directional bias, and its neutrality reaches the maximum value of $ 1 $.  

