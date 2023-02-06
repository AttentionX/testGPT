# learnGPT - week1

## Requirements

### Python version
```bash
python --version
```
```text
Python 3.9.12
```
### Libraries used
```bash
pip3 install torch==1.13.1 pytest==7.2.1 pyyaml==6.0
```

## TODO's
### TODO 1 - a naive LM (`GPTVer1`)

```bash
pytest tests/test_1.py -s -vv
```


### TODO 2 - taking the past into account  (`HeadVer1` & `GPTVer2`)

```bash
pytest tests/test_2.py -s -vv
```


### TODO 3 - vectorizing for loops (`HeadVer2`)

```bash
pytest tests/test_3.py -s -vv
```

### TODO 4 - taking the past into account with masking & normalization (`HeadVer3`)

```bash
pytest tests/test_4.py -s -vv
```

### TODO 5 - self-attention mechanism (`HeadVer4` & `GPTVer2`)

```bash
pytest tests/test_5.py -s -vv
```

### TODO 6 - positional encodings (`GPTVer3`)

```bash
pytest tests/test_6.py -s -vv
```





