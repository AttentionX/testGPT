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
    assert editdistance.eval("The berm,\nSXro sick's q-etcichors ", was) < 5
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1330s) | [Us (Korean) 🗣 Sounho](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c84eb74d-af76-47fd-9c78-0aa6a1d1c94f/Test1_Korean_AdobeExpress.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T152120Z&X-Amz-Expires=86400&X-Amz-Signature=32a1aa616ec869effe0ad969e55b6634761fc7ebba6aed31613e3673c1100a8c&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Adam](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/235b98a1-f465-40ac-8c67-1d9c2bfff3b0/Bigram_Language_Models.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T143825Z&X-Amz-Expires=86400&X-Amz-Signature=59ba8ba7dbfd30a299dc08a005d22510666698a17db2100e49aa4c79351797b4&X-Amz-SignedHeaders=host&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551321-0e83705d-3e08-439a-90c6-7eecf749f9f2.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553011-2c768eda-1ff7-46f0-bb64-127eb234449f.png">    | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224552980-8ca2d2b1-13f0-441b-b7c5-6a23104dac09.png"> | 

### contributors:

[Soun Ho Jung](https://github.com/aschung01) | [Adam Lee](https://github.com/Abecid)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/72554932?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/41438776?v=4">

# Test 2

```python 
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
    assert editdistance.eval("The oo rmnt oedi srnvhe\nd oy  phou", was) < 5


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
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s) | [Us (Korean) 🗣 Yuna](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e266f78a-5fbe-4cb8-a5db-d1bd378a086a/test2_ko.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T154914Z&X-Amz-Expires=86400&X-Amz-Signature=43d6d5a14a14408ec8030ad26114904a55317968c502bca7daa7ccb1872e4e60&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Junseon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/91309404-7bb7-4895-986e-ad71c259a0e3/test2.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144017Z&X-Amz-Expires=86400&X-Amz-Signature=7f37550972c1494ce0bf610d3720aff94a7e1175149fdeb38ba24995dcf2ff57&X-Amz-SignedHeaders=host&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551325-46e99bb7-5561-4486-a622-2f31d8f12e9e.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224556069-f31baadb-80a1-433a-8b81-b5c1d2038595.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553026-48559b0e-fb79-405a-8af1-7123aa5eda1e.png"> | 

### contributors:

[Yuna Park](https://github.com/gyuuuna) | [Jun Seon Kim](https://github.com/Junseon00)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/86581647?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/26458224?v=4">

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

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2831s) | [Us (Korean) 🗣 Junseon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c63d812c-cdad-4b81-9bf7-0d0be16e1b2a/test3_%E1%84%80%E1%85%B5%E1%86%B7%E1%84%8C%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%AB_AdobeExpress.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T151726Z&X-Amz-Expires=86400&X-Amz-Signature=417c87b2a875d78b8cb6534c3cdbb256d9c59d67a474b04bb83c152947e35e66&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Sounho](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/80c4ccc6-5fd0-4643-9cf8-217fce28becd/Test3_english.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144250Z&X-Amz-Expires=86400&X-Amz-Signature=b21bf4413cadcb52b70afe6485756ac13db548aa35d1cf4b3c5dd7bb74007cce&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Test3_english.mp4%22&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551328-15d656a2-c1df-41e4-adce-73f1a867842c.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553013-9eae140e-107c-49e0-8ac0-a6f423be2c4c.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553005-6379cb0c-7467-4d7d-9545-f996a9095168.png"> | 

### contributors:

[Jun Seon Kim](https://github.com/Junseon00) | [Soun Ho Chung](https://github.com/aschung01)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/26458224?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/72554932?v=4">

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
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer3()
    head(x)
    expected = torch.ones(B, T)
    was = head.wei.sum(dim=-1)
    assert torch.allclose(expected, was)


def test_head_v3_logits_are_properly_masked():
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
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3282s) | [Us (Korean) 🗣 Junyoung](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9d1851ab-3efe-4add-8967-e2bbdcee12bf/test4_한국어_박준영.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144409Z&X-Amz-Expires=86400&X-Amz-Signature=0072d603fc70217efe079cbb915495c77a06b1752e18573b569809ca3521cd1f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22test4_%25ED%2595%259C%25EA%25B5%25AD%25EC%2596%25B4_%25EB%25B0%2595%25EC%25A4%2580%25EC%2598%2581.mp4%22&x-id=GetObject)  | [Us (English) 🗣 Yuri](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c957a337-cda4-4c91-b61d-8d00f92b320b/TestGPT_4_eng.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144444Z&X-Amz-Expires=86400&X-Amz-Signature=bb16ad103d1ed0190286659c4b7a6ea9297dadcfd725185ef3a59d02429e80dc&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22TestGPT_4_eng.mp4%22&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551333-6c0bc6c0-2be6-45c0-9084-aab74756fafa.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224552982-12ddb0cd-63a6-4eb4-aa7c-30c1f8d17e5a.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553042-93bbee5f-4271-41ec-80b4-068f01420ced.png"> | 

### contributors:

[JunYoung Park](https://github.com/engineerA314) | [Yuri Kim](https://github.com/yuridekim)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/86403521?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/63735383?v=4">

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
    assert editdistance.eval("The st ano cmin he stesfveeman eco", was) < 5
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) | [Us (Korean) 🗣 Juhwan](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a925c698-ade0-4c14-8851-2071e980e767/Video_167b3cb8-74af-4637-8d89-c66f9138983b.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144544Z&X-Amz-Expires=86400&X-Amz-Signature=4ab4edc988111c7a3f6e66b8520cb5477231a1e3ccdacb9414dcb9d322211400&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Adam](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8dc1edd9-3b21-4b35-a782-8abb0aa451b3/self-attention.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T144626Z&X-Amz-Expires=86400&X-Amz-Signature=5f7f04093091393faf6a49016fc5b79e9d11446df374ebb6a5c89a66d7458bad&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22self-attention.mov%22&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551337-92ff1331-1a8e-4df4-ae5a-2be0dc79bf11.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553031-0ec32ab1-bf16-42d4-b4ba-54a1f2eb0f0e.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553014-284aeaa1-f5a5-4acf-95d0-424b70bebb1a.png"> | 

### contributors:

[Ju Hwan Cho](https://github.com/juhwancho) | [Adam Lee](https://github.com/Abecid)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49906112?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/41438776?v=4">

# Test 6

```python
def test_gpt_v1_and_v2_logits_order_is_not_preserved():
    """
    Ver1 & Ver2; You love that == That love you
    """
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    model = GPTVer2(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :], atol=1e-6, rtol=0.001)
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :], atol=1e-6, rtol=0.001)


def test_gpt_v3_pos_encodings_v1():
    """
    PE(w_pos) = w_pos
    """
    T, C = 4, 512
    # the first version of PE satisfies this property
    encodings = GPTVer3.pos_encodings_v1(T, C)
    short_encodings = GPTVer3.pos_encodings_v1(50, C)
    long_encodings = GPTVer3.pos_encodings_v1(100, C)
    # --- property 1 --- #
    # each position must be different
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- # (THIS DOES NOT HOLD)
    # values must not be too big to prevent gradient explosion
    with pytest.raises(AssertionError):
        assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    # time delta must be the same within a sentence.
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    # time delta must be the same across sentences of variable lengths.
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v2():
    """
    PE(w_pos) - w_pos / length
    """
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings_v2(T, C)
    short_encodings = GPTVer3.pos_encodings_v2(50, C)
    long_encodings = GPTVer3.pos_encodings_v2(100, C)
    # --- property 1 --- #
    # each position must be different
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    # values must not be too big to prevent gradient explosion
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    # time delta must be the same within a sentence.
    with pytest.raises(AssertionError):  # (THIS DOES NOT HOLD)
        assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                              torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    # time delta must be the same across sentences of variable lengths.
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v3():
    """
    PE(w_pos) = sin(w_pos / 10000^(i/C))
    """
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings_v3(T, C)
    short_encodings = GPTVer3.pos_encodings_v3(50, C)
    long_encodings = GPTVer3.pos_encodings_v3(100, C)
    # --- property 1 --- #
    # each position must be different
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    # values must not be too big to prevent gradient explosion
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    # time delta must be the same within a sentence.
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- # (THIS DOES NOT HOLD)
    # time delta must be the same across sentences of variable lengths.
    with pytest.raises(AssertionError):
        assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v4():
    """
    sinusoidal version of position encoding.
    PE(w_pos) = sin(w_pos / 10000^(i/C)) if i is even
    PE(w_pos) = cos(w_pos / 10000^((i)/C)) if i is odd
    should satisfy all properties
    """
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings_v4(T, C)
    short_encodings = GPTVer3.pos_encodings_v4(50, C)
    long_encodings = GPTVer3.pos_encodings_v4(100, C)
    # --- property 1 --- #
    # each position must be different
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    # values must not be too big to prevent gradient explosion
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    # time delta must be the same within a sentence.
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    # time delta must be the same across sentences of variable lengths.
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_logits_order_is_preserved():
    """
    As opposed to GPTVer1 & GPTVer2,  GPTVer3 preserves the order of the input tokens.
    e.g. You love that != That love you
    """
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
    With much to positional info, GPTVer3 picks up the Shakespearean pause.
    # --- from input.txt --- #
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
    assert editdistance.eval("The t weou fedothtotoutho,\nI- Iowh", was) < 5
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3618s) | [Us (Korean) 🗣 Hahyeon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/be1a5046-27ea-4d65-b1e5-888231db51be/Positional_Encoding.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T155723Z&X-Amz-Expires=86400&X-Amz-Signature=04f59b5a96141403c90e539568b6b00b17c3addfc837437ebfcb06f84c789e4d&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Eubin]() |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551315-108ee008-7c41-4893-abac-00be63a4411a.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224556357-75f08aee-e8fc-4bb4-8af0-82121c800820.png">     | <img width="400" alt="image" src=""> | 

### contributors:

[Ha Hyeon Choi](https://github.com/hahyeon610) | [Eu-Bin KIM](https://github.com/eubinecto)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49430956?s=400&v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/56193069?v=4">

# Test 7

```python
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
    # gpt should converge faster with multi-head
    assert losses_1['val'] > losses_multi['val']


def test_head_ver_4_and_multi_head_ver_1_are_equally_expensive():
    """
    (Vaswani et al. 2017, Attention is all you need)
    "Due to the reduced dimension of each head,
    the total computational cost is similar to that of single-head attention with full dimensionality."
    dk = dv = dmodel/h, where h = number of heads.
    trainable parameters of multi-head ver 1 and head ver 4 must be the same because
    head_size = embed_size // n_heads
    """
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    head_v4 = HeadVer4(T, C, C)
    assert sum([p.numel() for p in multi_head_v1.heads.parameters() if p.requires_grad]) \
           == sum([p.numel() for p in head_v4.parameters() if p.requires_grad])


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


[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s) | [Us (Korean) 🗣 Hahyeon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4a5eac33-b2d0-49a3-b99c-ccbadc38c9d7/MultiHead.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T161144Z&X-Amz-Expires=86400&X-Amz-Signature=ec0f2da2019d6fdb1d38d8fff86cc0d70fa65cc56197f01703e164351846f869&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22MultiHead.mp4%22&x-id=GetObject)  | [Us (English) 🗣 Eubin](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4753b6df-af7f-4c9b-b418-72e80391083a/video1515235742_exported_no_caption.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T145119Z&X-Amz-Expires=86400&X-Amz-Signature=6cc56f9e5983059590849299ede37a27dcbcd3c04052e5917286f21b580b4f5b&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22video1515235742_exported_no_caption.mp4%22&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551322-f1b845e7-c6e1-4022-bbd3-3673ae6413a9.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224557207-0b04b155-eec0-4120-aa37-0b7b6c826969.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553049-cb9ed874-6373-412b-b5f7-2dbfdf9a8e25.png"> | 

### contributors:

[Ha Hyeon Choi](https://github.com/hahyeon610) | [Eu-Bin KIM](https://github.com/eubinecto)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49430956?s=400&v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/56193069?v=4">

# Test 8

```python
def test_ffn_helps():
    """
    GPT converges faster with ffn.
    """
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
    """
    Deep converges faster with residual connection.
    """
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


[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s) | [Us (Korean) 🗣 Yuri](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f3dd00e2-801f-4a0f-bc15-f1fd5c4fcdf7/Test8_kor.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T145247Z&X-Amz-Expires=86400&X-Amz-Signature=08e03ecabed1f1ebff9766ef3e688f25c8413b7e0bdb26b591ad90c87ee162eb&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Juhwan](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e9ff9c44-50fc-42bd-b708-45aaffd36b5f/Video_4b041d25-eec5-4c92-9f33-6b15ab096855.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T145249Z&X-Amz-Expires=86400&X-Amz-Signature=67aeaf43872e040315c96c9946923f3bcfc8aa59457152e97fc94c3afe5322c3&X-Amz-SignedHeaders=host&x-id=GetObject) |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551326-725c9581-1ce4-4de8-a434-5651710c049e.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553033-5b8485c9-6fee-4756-99bf-3157ff9ac7db.png">     | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224553020-9fd6a2e4-4a4e-451d-b783-88f756be52a3.png"> | 

### contributors:

[Yuri Kim](https://github.com/yuridekim) | [Ju Hwan Cho](https://github.com/juhwancho)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/63735383?v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49906112?v=4">

# Test 9

```python
# test: gpt v4 learns faster with LayerNorm
def test_layer_norm_helps_when_network_is_deep():
    """
    layer norm helps because it mitigates vanishing gradient.
    """
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


def test_layer_norm_mitigates_vanishing_gradient():
    """
    a simple experiment to see if layer norm mitigates vanishing gradient.
    """
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


def test_layer_norm_properly_normalizes_the_feature_dimension():
    """
    Layenorm mitigates vanishing gradient by normalizing the features dimension.
    """
    B, T, C = 32, 64, 512
    ln = LayerNorm(C)
    x = torch.randn(T, C)
    out = ln(x)
    mean_across_features = torch.round(out.mean(dim=-1))
    var_across_features = torch.round(out.var(dim=-1))
    assert torch.allclose(mean_across_features, torch.zeros(mean_across_features.shape))
    assert torch.allclose(var_across_features, torch.ones(var_across_features.shape))
```

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s) | [Us (Korean) 🗣 Hahyeon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a1fa2aba-2cac-40f1-b68c-66c6a240fd2d/LayerNorm.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T160259Z&X-Amz-Expires=86400&X-Amz-Signature=d127185606c97c3bb05b37fd3502816f5ca28c0bdbd88b6305470a9b348f2183&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22LayerNorm.mp4%22&x-id=GetObject)  | [Us (English) 🗣 Eubin]() |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551329-5872391e-2729-467c-87e7-da9f0cef73a2.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224556884-6bc1d2a7-089a-4ba3-af9a-5f5c4feded72.png">     | <img width="400" alt="image" src=""> | 

### contributors:

[Ha Hyeon Choi](https://github.com/hahyeon610) | [Eu-Bin KIM](https://github.com/eubinecto)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49430956?s=400&v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/56193069?v=4">

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
    dropout helps because it mitigates overfitting.
    """
    seed_everything(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    #  --- push the model to overfit --- #
    train_ratio = 0.001
    n = int(len(conftest.data) * train_ratio)
    conftest.train_data = conftest.data[:n]
    conftest.val_data = conftest.data[n:]
    config['max_iters'] = 10000
    config['learning_rate'] = 0.005
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

[Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5864s) | [Us (Korean) 🗣 Hahyeon](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f33e78f8-8c16-4bc2-b903-6081423a8580/Dropout.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230312T155950Z&X-Amz-Expires=86400&X-Amz-Signature=c805d985597ae09d2a5134aede22abcf7ae46c4c2e9d58127a4ef79ac7af2350&X-Amz-SignedHeaders=host&x-id=GetObject)  | [Us (English) 🗣 Eubin]() |
--- |---------| --- |
<img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224551335-5ed1a6d0-cdbf-4d4a-a771-f1578085f233.png"> | <img width="400" alt="image" src="https://user-images.githubusercontent.com/49430956/224556689-d7623fcd-fcb9-4e41-959c-5e5910938b4e.png">     | <img width="400" alt="image" src=""> | 

### contributors:

[Ha Hyeon Choi](https://github.com/hahyeon610) | [Eu-Bin KIM](https://github.com/eubinecto)
--- | --- 
<img width="100" alt="image" src="https://avatars.githubusercontent.com/u/49430956?s=400&v=4"> | <img width="100" alt="image" src="https://avatars.githubusercontent.com/u/56193069?v=4">
