## 1. Bigram Language Model (`GPTVer1`)
1. **Create a naive bigram language model**
    1. Get the predictions (logit)
    2. Apply softmax on the last prediction
    3. Sample from the distribution
    4. Concat to the context
2. Files to Edit
    - gpt_v1.py
3. Test Cases
    - Check that the the model output is correct
```bash
pytest tests/test_1.py -s -vv
```

---

## 2. N-gram Language Model (For Loops)  (`HeadVer1` & `GPTVer2`)
1. **Take the previous tokens into account with the mean**
    1. Using for loops, return the mean of the context in the model head
    2. Create the logits via the token embeddings, the model head, and a linear projection layer
2. Files to Edit
    1. head_v1.py
    2. gpt_v2.py
3. Test Cases
    1. Check that the model head returns the mean of the previous context
    2. Check that the model output is correct
```bash
pytest tests/test_2.py -s -vv
```

---

## 3. N-gram Language Model (Tensor Multiplications) (`HeadVer2`)
1. **Calculate the mean of the context with tensor multiplications**
    1. Create a triangular mask for autoregressive training
    2. Divide by sum for calculating the proportion for each token
    3. Tensor multiplication with x to calculate the mean
2. Files to Edit
    - head_v2.py
3. Test Cases
    1. Check that the means from HeadVer1 and HeadVer2 are the same
    2. Check that the time taken for calculating the mean for HeaVer2 is smaller than HeadVer1

```bash
pytest tests/test_3.py -s -vv
```

---

## 4. Masking & Softmax (`HeadVer3`)
1. **Calculate the attention score via masking and softmax**
    1. Create a triangular mast for autoregressitve training
    2. Converts 0s with '-inf'
    3. Calculate the softmax for each context
    4. Matrix multiplication with x
2. Files to Edit
    - head_v3.py
3. Test Cases
    1. Check that the means from HeadVer1 and HeadVer3 are the same (allclose)
    2. Check that the speed for calculating HeavVer3 is faster than HeadVer1
    3. Check that the logits are properly normalized
    4. Check that the logits are properly masked


```bash
pytest tests/test_4.py -s -vv
```

---

## 5. Self-attention mechanism (`HeadVer4` & `GPTVer2`)
1. **Implement Self-Attention**
    1. Calculate the attention weights via a dot product with q and k with the scale C
    2. Apply softmax
    3. Tensor multiplication with v
2. Files to Edit
    - head_v4.py
3. Test Cases
    1. Check that the attention scores of two tensors that are in different order are the same.
    2. Check that the outputs are properly masked
    3. Check that the outputs are properly normalized
    4. Check that the the model output is correct
```bash
pytest tests/test_5.py -s -vv
```

---

## 6. Positional encodings (`GPTVer3`)
1. ***Imlement Positional Encoding**
    1. Positional Encoding
    2. Logits
        1. Get the token embeddings
        2. Add the embeddings with the positional encoding
        3. Run the output through the attention head
        4. Run the output through the model head
2. Files to Edit
    - gpt_v3.py
3. Test Cases
    1. Check that the orders are preserved
    2. Check that the positional difference stays constant
    3. Check that the orders are preserved in the logits
    4. Check that the the model output is correct
```bash
pytest tests/test_6.py -s -vv
```

---

## 7. Multi-head attention (`MultiHeadVer1`)

```shell
pytest tests/test_7.py -s -vv
```

### test_head_ver_4_and_multi_head_ver_1_are_equally_expensive & test_multi_head_helps

<img src='img/Multi-Head Attention.png' width=250>

Based on the `HeadVer4` (self-attention head) implemented in Week1, we will implement multi-head attention.  
After passing through the FC layers, Q, K, and V in the self-attention head change shape from (batch_size, block_size, embed_size) → (batch_size, block_size, head_size)  
And the relationship embed_size = head_size * n_heads holds.

> TODO 1: Implement `MultiHeadVer1.forward`.
> Pass the input x through n_heads self-attention heads and then concatenate the head outputs. Finally, pass the concatenated output through a projection layer (FC) to implement multi-head attention.

Please run the tests and answer the following questions:
1. Is there a difference in computational cost between `MultiHeadVer1` and `HeadVer4`? If not, why?
2. Which one performs better, `MultiHeadVer1` or `HeadVer4`? Why?

### test_multi_head_ver_2_is_faster_than_ver_1 & test_multi_head_ver_1_and_multi_head_ver_2_are_logically_equal

Please run the tests and answer the following questions:
1. `MultiHeadVer2` and `MultiHeadVer1` have no difference in algorithm, but `MultiHeadVer2` is faster in computation. Why is that?
---

## 8. Feed forward network & Residual connections (`MultiHeadVer2`)

```shell
pytest tests/test_8.py -s -vv
```

### test_ffn_helps

<img src='img/BlockVer1.png' width=250>

`BlockVer1` performs Multi-Head Attention and FeedForward as implemented above.  

> TODO 1: Implement the `FeedForward`
>     1. $FFNN(x)=Max(0, W_1x+b_1)W_2 + b_2$
>     2. $W_1$ has shape (embed_size, 4 $\times$ embed_size), $W_2$ has shape (4 $\times$ embed_size, embed_size).  

> TODO 2: Implement `BlockVer1.forward`
>    1. It should pass through Multi-Head Attention and then through the FeedForward layer.  

Please run the tests and answer the following questions:  
1. Which one performs better, `MultiHeadVer2` or `BlockVer1`? Why?


### test_residual_conn_helps_when_network_is_deep

<img src='img/BlockVer2.png' width=250>

`BlockVer2` is a block that adds residual connection to `BlockVer1`.  

> TOdO 3: Implement `BlockVer2.forward`
>   1. You should add a residual connection to each of Multi-Head Attention and FeedForward.


Please run the tests and answer the following questions:
1. Which one performs better, `BlockVer1` or `BlockVer2`? Why?

---

## 9. Layer Normalization (`BlockVer3`)

```shell
pytest tests/test_9.py -s -vv
```

### test_layer_norm_features_dim_is_properly_normalized & test_layer_norm_mitigates_vanishing_gradient


> TODO 1: Please implement `LayerNorm` to be used in `BlockVer3`.

Please run the tests and answer the following questions:
1. What are the characteristics after passing through `LayerNorm`?
2. How does `LayerNorm` alleviate the vanishing gradient problem?

### test_layer_norm_helps_when_network_is_deep

<img src='img/BlockVer3.png' width=250>

> TODO 2: Implement `BlockVer3.forward`. Add `LayerNorm` to the inputs of Multi-Head and FeedForward as shown in the figure above.

Please run the tests and answer the following questions:
1. Which one performs better, `BlockVer2` or `BlockVer3`? Why?

--- 

## 10. Dropout (`BlockVer4`)

```shell
pytest tests/test_10.py -s -vv
```
### test_block_ver_4_output_is_always_different_in_train_mode & test_block_ver_4_output_is_always_the_same_in_eval_mode

<img src='img/BlockVer4.png' width=250>

> TODO 4: Please implement `BlockVer4.forward`. You should add a dropout layer to the outputs of Multi-Head and FeedForward, a total of 2 places.

Please run the tests and answer the following questions:
1. Why is the output of `BlockVer4` always different in training mode?
2. Why is the output of `BlockVer4` always the same in evaluation mode?

### test_dropout_helps

Please run the tests and answer the following questions:
1. Which one performs better, `BlockVer3` or `BlockVer4`? Why?

