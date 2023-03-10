## 1. Bigram 언어 모델 (`GPTVer1`)
1. **단순한 biram 언어모델을 구현하세요**
    1. Get the predictions (logit)
    2. Apply softmax on the last prediction
    3. Sample from the distribution
    4. Concat to the context
2. Files to Edit
    - gpt_v1.py
3. Test Cases
    - Get the same output as the test case
```bash
pytest tests/test_1.py -s -vv
```

---

## 2. Taking the past into account  (`HeadVer1` & `GPTVer2`)

```bash
pytest tests/test_2.py -s -vv
```

---

## 3. Matrix multiplication instead of for loops (`HeadVer2`)

```bash
pytest tests/test_3.py -s -vv
```

---

## 4. Masking & Normalization (`HeadVer3`)

```bash
pytest tests/test_4.py -s -vv
```

---

## 5. Self-attention mechanism (`HeadVer4` & `GPTVer2`)

```bash
pytest tests/test_5.py -s -vv
```

---

## 6. Positional encodings (`GPTVer3`)

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

Week1에서 구현했던 `HeadVer4`(self-attention head)를 바탕으로 multi-head attention을 구현합니다.
self-attention head에서 Q, K, V가 각각 FC layer를 통과하고나면 (batch_size, block_size, embed_size) → (batch_size, block_size, head_size)로 shape이 변경이 됩니다.
그리고 embed_size = head_size * n_heads의 관계가 성립합니다.

> TODO 1: `MultiHeaVer1.forward`를 구현해주세요.
> input x를 n_heads 개의 self-attention head를 통과한 후 head_output을 concatnate합니다. 그리고 projection layer(FC)를 통과시켜 multi-head attention을 구현해주세요.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer1`와 `HeadVer4`의 연산량에 차이가 있나요? 없다면 왜?
2. `MultiHeadVer1`와 `HeadVer4` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

### test_multi_head_ver_2_is_faster_than_ver_1 & test_multi_head_ver_1_and_multi_head_ver_2_are_logically_equal

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer2`와 `MultiHeadVer1`는 알고리즘에 차이는 없지만 `MultiHeadVer2`가 연산 속도가 더 빠릅니다. 그 이유는 무엇인가요?
---

## 8. Feed forward network & Residual connections (`MultiHeadVer2`)

```shell
pytest tests/test_8.py -s -vv
```

### test_ffn_helps

<img src='img/BlockVer1.png' width=250>

`BlockVer1`은 위에서 구현한 Multi-Head Attention과 FeedForward를 수행합니다.

> TODO 2-1: `FeedForward`를 구현해주세요. $FFNN(x)=Max(0, W_1x+b_1)W_2 + b_2$  
> ($W_1$의 shape은 (embed_size, 4 $\times$ embed_size), $W_2$의 shape은 (4 $\times$ embed_size, embed_size) 입니다.)

> TODO 2-2: `BlockVer1.forward`을 구현해주세요. Multi-Head Attention을 통과한 뒤 FeedForward layer를 통과시키면 됩니다.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer2`와  `BlockVer1` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?


### test_residual_conn_helps_when_network_is_deep

<img src='img/BlockVer2.png' width=250>

`BlockVer2`는 BlockVer1에서 Residual connection을 추가한 Block입니다.

> TODO 2-3: `BlockVer2.forward`를 구현해주세요. Multi-Head Attention과 FeedForward에 대해 각각 residual Connection을 추가하면 됩니다.


테스트를 돌려보고 다음의 질문에 답해주세요.
1. `BlockVer1`과 `BlockVer2` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

---

## 9. Layer Normalization (`BlockVer3`)

```shell
pytest tests/test_9.py -s -vv
```

### test_layer_norm_features_dim_is_properly_normalized & test_layer_norm_mitigates_vanishing_gradient


> TODO 3-1: `BlockVer3`에서 사용할 `LayerNorm`을 구현해주세요.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `LayerNorm`을 통과한 후에는 어떤 특징이 있나요?
2. `LayerNorm`은 어떻게 기울기 소실 문제를 완화하나요?

### test_layer_norm_helps_when_network_is_deep

<img src='img/BlockVer3.png' width=250>

> TODO 3-2: `BlockVer3.forward`를 구현해주세요. LayerNorm을 위의 그림처럼 Multi-Head의 input, FeedForward의 input 총 2곳에 추가하시면 됩니다.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `BlockVer2`와 `BlockVer3` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

--- 

## 10. Dropout (`BlockVer4`)

```shell
pytest tests/test_10.py -s -vv
```
### test_block_ver_4_output_is_always_different_in_train_mode & test_block_ver_4_output_is_always_the_same_in_eval_mode

<img src='img/BlockVer4.png' width=250>

> TODO 4: `BlockVer4.forward`를 구현해주세요. Dropout layer를 추가하면됩니다. Dropout은 Multi-Head의 output, FeedForward의 output 총 2곳에 추가하시면 됩니다.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. 왜 학습 모드 일 때 `BlockVer4`의 출력은 항상 다를까요?
2. 왜 평가 모드 일 때 `BlockVer4`의 출력은 항상 같을까요?

### test_dropout_helps

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `BlockVer3`와 `BlockVer4` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

