# testGPT

<img width="1059" alt="image" src="https://user-images.githubusercontent.com/56193069/216820890-187634ee-1058-4d99-83e6-cf4173817002.png">

# Test 1

```python3
def test_gpt_v1_generates_text_given_a_context():
    """
    Note how gpt v1 ends the sentence rather abruptly.
    """
    seed_everything(1337)
    lm = GPTVer1(config['vocab_size'], config['block_size'])
    train(lm)
    was = generate(lm, "The ", 30)
    expected = "The \nSXro sick's q-etcichors "
    assert expected == was
```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 


# Test 2

```python 

def test_head_v1_takes_an_average_of_the_past_into_account():
    """
    implement: TODO 2 - 1
    """
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    expected = torch.Tensor([[[1,   2,   3  ],
                              [2.5, 3.5, 4.5],
                              [4,   5,   6  ]]])
    head = HeadVer1()
    was = head(x)
    assert torch.allclose(expected, was)


def test_gpt_v2_and_head_v1_generates_text_given_a_context():
    """
    implement: TODO 2 - 2
    """
    seed_everything(1337)
    head = HeadVer1()
    V, T, C = config['vocab_size'], config['block_size'], config['embed_size']
    lm = GPTVer2(head, V, T, C)
    train(lm)  # may take a while
    was = generate(lm, "The ", 30)
    expected = "The oo rmnt oedi srnvhe\nd oy  phou"
    assert expected == was

```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 

# Test 3

```python
def test_head_v2_and_head_v1_are_logically_identical():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head_v1 = HeadVer1()
    head_v2 = HeadVer2()
    y_v1 = head_v1(x)
    y_v2 = head_v2(x)
    assert torch.allclose(y_v1, y_v2)


def test_head_v2_faster_than_head_v1():
    x = torch.rand(4, 128, 1024)
    head_v1 = HeadVer1()
    head_v2 = HeadVer2()
    time_taken_v1 = timeit.timeit(lambda: head_v1(x), number=10)
    time_taken_v2 = timeit.timeit(lambda: head_v2(x), number=10)
    assert time_taken_v2 < time_taken_v1

```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 


# Test 4

```python
def test_head_v3_and_head_v1_are_logically_identical():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head_v1 = HeadVer1()
    head_v3 = HeadVer3()
    y_v1 = head_v1(x)
    y_v3 = head_v3(x)
    assert torch.allclose(y_v1, y_v3)


def test_head_v3_faster_than_head_v1():
    x = torch.rand(4, 128, 1024)
    head_v1 = HeadVer1()
    head_v3 = HeadVer3()
    time_taken_v1 = timeit.timeit(lambda: head_v1(x), number=10)
    time_taken_v3 = timeit.timeit(lambda: head_v3(x), number=10)
    assert time_taken_v3 < time_taken_v1


def test_head_v3_logits_are_properly_normalized():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head = HeadVer3()
    head(x)
    expected = torch.IntTensor([[[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)


def test_head_v3_logits_are_properly_masked():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer3()
    head(x)
    expected = torch.ones(B, T)
    was = head.wei.sum(dim=-1)
    assert torch.allclose(expected, was)

```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 



# Test 5

```python
def test_head_v4_attention_has_no_notion_of_space():
    """
    :return:
    """
    x1 = torch.Tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
    x2 = torch.Tensor([[[4, 5, 6],
                        [1, 2, 3],
                        [7, 8, 9]]])
    _, T, C = x1.shape
    head = HeadVer4(T, C, C)
    y1 = head(x1)  # (B, T, C)
    y2 = head(x2)  # (B, T, C)
    assert torch.allclose(y1[:, -1, :], y2[:, -1, :])


def test_head_v4_logits_are_properly_masked():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    T, C = x.shape[1:]
    head = HeadVer4(T, C, C)
    head(x)
    expected = torch.IntTensor([[[0,  1,  1],
                                 [0,  0,  1],
                                 [0,  0,  0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)


def test_head_v4_logits_are_properly_normalized():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer4(T, C, C)
    head(x)
    expected = torch.ones(B, T)
    was = head.wei.sum(dim=-1)
    assert torch.allclose(expected, was)


def test_head_v4_the_variance_of_wei_after_scale_is_1():
    B, T, C = 4, 128, 1024
    x = torch.randn(B, T, C)
    head = HeadVer4(T, C, C)
    head(x, test=True)  # (B, T, C)
    assert 1 == torch.round(head.var)


def test_gpt_v2_and_head_v4_generates_text_given_a_context():
    seed_everything(1337)
    V, T, C = config['vocab_size'], config['block_size'], config['embed_size']
    head = HeadVer4(T, C, C)
    lm = GPTVer2(head, V, T, C)
    train(lm)  # may take a while
    was = generate(lm, "The ", 30)
    expected = "The st ano cmin he stesfveeman eco"
    assert expected == was

```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 


# Test 6

```python

def test_gpt_v1_logits_order_is_not_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v2_logits_order_is_not_preserved():
    torch.manual_seed(1337)
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer2(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_pos_encodings_each_pos_is_different():
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    assert not torch.allclose(encodings[2], encodings[3])


def test_gpt_v3_pos_encodings_dist_stays_constant():
    T, C = 10, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))
    assert torch.allclose(torch.norm(encodings[5] - encodings[3]), torch.norm(encodings[6] - encodings[4]))
    assert torch.allclose(torch.norm(encodings[7] - encodings[5]), torch.norm(encodings[8] - encodings[6]))


def test_gpt_v3_logits_order_is_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert not torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert not torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert not torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_and_head_v4_generates_text_given_a_context():
    """
    with positional encodings added, gpt picks up Shakespearean pause (comma), so to speak.
    e.g. We are accounted poor citizens, the patricians good.
    e.g. Let us kill him, and we'll have corn at our own price.
    e.g. I say unto you, what he hath done famously, he did
    """
    seed_everything(1337)
    V, T, C = config['vocab_size'], config['embed_size'], config['block_size']
    head = HeadVer4(T, C, C)
    lm = GPTVer3(head, V, T, C)
    train(lm)  # may take a while
    was = generate(lm, "The ", 30)
    expected = "The t weou fedothtotoutho,\nI- Iowh"
    assert expected == was


```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 


# Test 7

```python
def test_head_ver_4_and_multi_head_ver_1_are_equally_expensive():
    """
    trainable parameters of multi-head ver 1 and head ver 4 must be the same because
    head_size = embed_size // n_heads
    """
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    head_v4 = HeadVer4(T, C, C)
    assert sum([p.numel() for p in multi_head_v1.heads.parameters() if p.requires_grad]) \
           == sum([p.numel() for p in head_v4.parameters() if p.requires_grad])


def test_multi_head_helps():
    """
    But multi-head leads to faster convergence than single head.
    """
    seed_everything(1337)
    V, T, C, n_heads = config['vocab_size'], config['block_size'], config['embed_size'], config['n_heads']
    # --- HeadVer4: single-head --- #
    contextualizer = HeadVer4(T, C, C)
    gpt = GPTVer4(contextualizer, V, T, C)
    losses_1 = train(gpt)
    # --- MultiHeadVer4: multi-head --- #
    contextualizer = MultiHeadVer1(T, C, n_heads)
    gpt = GPTVer4(contextualizer, V, T, C)
    losses_multi = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['val'] > losses_multi['val']


def test_multi_head_ver_2_is_faster_than_ver_1():
    """
    MultiHeadVer2 is faster than MultiHeadVer1 because it does not involve explicit loops.
    """
    B, T, C = 32, 64, 512
    n_heads = 8
    x = torch.randn(B, T, C)
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    time_taken_v1 = timeit.timeit(lambda: multi_head_v1(x), number=10)
    time_taken_v2 = timeit.timeit(lambda: multi_head_v2(x), number=10)
    assert time_taken_v2 < time_taken_v1


def test_multi_head_ver_1_and_multi_head_ver_2_are_logically_identical():
    """
    And they are logically identical.
    """
    B, T, C = 1, 3, 8
    n_heads = 4
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    multi_head_v2.query.weight = torch.nn.Parameter(torch.concat([head.query.weight
                                                                  for head in multi_head_v1.heads], dim=0))
    multi_head_v2.key.weight = torch.nn.Parameter(torch.concat([head.key.weight
                                                                for head in multi_head_v1.heads], dim=0))
    multi_head_v2.value.weight = torch.nn.Parameter(torch.concat([head.value.weight
                                                                  for head in multi_head_v1.heads], dim=0))
    multi_head_v2.proj.weight = torch.nn.Parameter(multi_head_v1.proj.weight)
    multi_head_v2.proj.bias = torch.nn.Parameter(multi_head_v1.proj.bias)
    x = torch.randn(B, T, C)
    out_1 = multi_head_v1(x)
    out_2 = multi_head_v2(x)
    assert torch.allclose(out_1, out_2)






```


Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 



# Test 8

```python
def test_ffn_helps():
    seed_everything(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- MultiHeadVer2: multi-head --- #
    contextualizer = MultiHeadVer2(T, C, n_heads)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer1: multi-head + ffn --- #
    contextualizer = BlockVer1(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['val'] > losses_2['val']


def test_residual_conn_helps_when_network_is_deep():
    seed_everything(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- Layers of BlockVer1: multi-head + ffn --- #
    contextualizer = torch.nn.Sequential(*[BlockVer1(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- Layers of BlockVer2: multi-head + ffn + residual --- #
    contextualizer = torch.nn.Sequential(*[BlockVer2(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['val'] > losses_2['val']


```


Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 



# Test 9

```python

def test_layer_norm_features_dim_is_properly_normalized():
    B, T, C = 32, 64, 512
    ln = LayerNorm(C)
    x = torch.randn(T, C)
    out = ln(x)
    mean_across_features = torch.round(out.mean(dim=-1))
    var_across_features = torch.round(out.var(dim=-1))
    assert torch.allclose(mean_across_features, torch.zeros(mean_across_features.shape))
    assert torch.allclose(var_across_features, torch.ones(var_across_features.shape))


def test_layer_norm_mitigates_vanishing_gradient():
    depth = 1000
    B, T, C = 3, 64, 128
    x = torch.randn(B, T, C, requires_grad=True)

    # Measure gradients without LayerNorm
    without_norm = x.clone().detach().requires_grad_(True)
    for i in range(depth):
        outputs = torch.nn.Linear(C, C)(without_norm)
        outputs.sum().backward()
        without_norm = outputs.clone().detach().requires_grad_(True)
    # gradients should be near-zero
    assert torch.allclose(torch.round(without_norm), torch.zeros(without_norm.shape))

    # Measure gradients with LayerNorm
    with_norm = x.clone().detach().requires_grad_(True)
    for i in range(depth):
        outputs = torch.nn.Linear(C, C)(with_norm)
        outputs = LayerNorm(C)(outputs)
        outputs.sum().backward()
        with_norm = outputs.clone().detach().requires_grad_(True)
    # gradients should not be near-zero
    assert not torch.allclose(torch.round(with_norm), torch.zeros(with_norm.shape))


# test: gpt v4 learns faster with LayerNorm
def test_layer_norm_helps_when_network_is_deep():
    seed_everything(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- layers of BlockVer2: multi-head + ffn + residual --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer2(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- layers of BlockVer3: multi-head + ffn + residual + layer norm --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer3(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['val'] > losses_2['val']


```



Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 




# Test 10

```python
def test_block_ver_4_output_is_always_different_in_train_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.train()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert not torch.allclose(out_1, out_2)
    assert not torch.allclose(out_2, out_3)
    assert not torch.allclose(out_3, out_4)


def test_block_ver_4_output_is_always_the_same_in_eval_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.eval()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert torch.allclose(out_1, out_2)
    assert torch.allclose(out_2, out_3)
    assert torch.allclose(out_3, out_4)


def test_dropout_helps():
    """
    dropout mitigates overfitting.
    """
    seed_everything(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    # push the model to overfit
    config['max_iters'] = 7500
    config['learning_rate'] = 0.01
    # --- BlockVer3: layers of multi-head + ffn + residual + layer norm --- #
    contextualizer = BlockVer3(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer4: layers of multi-head + ffn + residual + layer norm + dropout --- #
    contextualizer = BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    # "mitigates overfitting" = train loss is bigger but validation loss is smaller.
    assert losses_1['train'] < losses_2['train']
    assert losses_1['val'] > losses_2['val']


```

Karpathy | Us (Korean)  | Us (English) |
--- |---------| --- |
... | ...     | ... | 





# Contributors

Eu-Bin KIM |  ... | ... |  
--- | --- | --- |
... | ... | ... |
