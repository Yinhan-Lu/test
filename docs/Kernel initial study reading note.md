# Kernel initial study reading note

**Weakness 1: The initial study primarily focuses on a RoBERTa. Evaluating it only demonstrates the influence in Encoder-Only Models.**

Transformer-based models have evolved into many specialized types depending on their architecture and use case, such as Decoder-only ones(GPT, etc) and Encoder-Decoder (Seq2Seq) ones(T5, etc). By evaluating the bias transfer in those specialized types, we could demonstrate more deeply how training data imbalances can influence cultural representations at the embedding level in transformer-based models.

Here is a simple how-to about applying the experiment on autoregressive model:

1. model choosing: smaller version of GPT(like GPT2) from HF
2. do finetuning using the same dataset as we do in the initial study
3. replacing MLM task to next token prediction
4. do same analysis in the paper.

**Weakness 2: missing the discussion about the relationship between the specific architecture of transformer-based models and bias transfer(Following up of Weakness 1)**

By seeing how different nature of the models influences these embeddings (how the autoregressive nature of the model influences these embeddings, etc) can not only demonstrate that the bias transfer in transformer based model more certainly(like we said in weakness 1), but also reveal at least these problems:

1. Whether different architecture leads to different types or intensities of cultural bias in the embeddings.
2. How the semantic relationships between culturally charged word pairs differ based on the model architecture
3. if architecture-specific mitigation strategies are needed.

Hence, on the basis of experiments with different architectures, We could have more discussion about the comparison between them.

**Weakness 3: Limited Number of Word Pairs:**

The initial study's embedding space analysis in section 5.1 and figure 1 shows the bias embedded in the models by the variation in cosine similarity scores for selected word pairs. However, we only evaluate only a few pairs of words. To generalize about biases in an entire embedding space, using only a few pairs is kind of statistically shallow and may not capture broader embedding trends about the change of cultural representations at the embedding level due to imbalanced training data.

I am still not sure how to build a fair comprehensive word pairs set fully automatically. However, even just use GPT-4o to generate hundreds of word pairs and check the fairness manually is still doable.

**Weakness 4: Lack of Debiasing method**

Even though it is mentioned in the initial study that the inclusion of the Nunavut Hansard corpus in fine-tuning data of transformer models is essential to mitigating bias towards the First Nations of Canada and the Inuit of Nunavut, we didn't try to mitigate the biases in depth. Then, we missed the opportunity to verify if biases are inherent or can be fixed by simple means and limited the practical value of this study. There are some doable technique that we can try on our case to mitigate biases, one is Iterative Nullspace Projection (INLP)(https://arxiv.org/pdf/2004.07667)

**Weakness 5: Treat the model as a black box**

The model are treated as a black box in the initial study. Firstly, We know that using different finetune data leads to different result in those MLM tasks. However, we don't know

1. the token-level cause of the difference.
2. the layer that the biases are encoded

This limits interpretability and the ability to target bias at its source. Here are some techniques that we can use to do layer-level and token-level analysis:

1. Layer-wise Probing: taking template with<mask> and replace the mask with different values like ("Indigenous", "Canadian"). retrieve the hidden state of each layer of the model and calculate cosine similarity as we do in the paper. This would show the layer that biases are constructed.
2. integrated gradient(https://arxiv.org/html/2406.10130v1#:~:text=try%20to%20unveil%20the%20mystery,inside%20PLM%20units%20to%20achieve): With IG, we would see which token in a sentence pushed the model towards a biased choice for the mask word.