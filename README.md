# reason

reason is a small ml experiment looking at how training noise and distribution shift affect model predictions, and whether prediction uncertainty actually reflects error.

the experiment uses a synthetic dataset with a known relationship:

```text
y = 2x
```

the relationship is used to generate the ground truth. the model does not receive the rule directly. it only gets the generated training examples and learns the relationship using linear regression.

## what the experiment tests

the experiment has three parts.

first, different amounts of noise are added to the training data. the noise levels are `0, 1, 2, 4, and 8`. each setting is run 100 times and the mean absolute error is recorded.

second, the test distribution is changed. training values are sampled from `1–20`. one set of test inputs stays within that range, while another uses `21–30`. this gives a simple comparison between inputs from the same range as the training data and inputs outside it.

third, prediction uncertainty is estimated by retraining the model on different noisy training samples. for each input, 100 predictions are collected and the spread of those predictions is used as an empirical measure of uncertainty.

this is not meant to be a formal confidence score or calibrated probability. it is simply a way to measure how much the model's prediction changes when the training data changes.

## results

the experiment produces three graphs.

`noise_results.png` shows the relationship between training noise and prediction error.

`model_error.png` compares error on normal inputs with error under distribution shift as the amount of training noise changes.

`uncertainty.png` compares the spread of repeated predictions with the actual prediction error for different inputs.

the results show that increasing training noise increases prediction error. the shifted inputs also have higher error than inputs from the original training range, with the difference becoming larger as the noise increases.

the uncertainty result is less direct. prediction spread changes as the input moves away from the training range, but it does not consistently track the actual error. so the uncertainty measure is not treated as a reliable indicator of error in this experiment.

## why synthetic data

synthetic data keeps the experiment controlled and gives a known ground truth.

the training range, test range, and amount of noise can be changed independently. this makes it easier to isolate what is causing a change in the model's behaviour without introducing other variables from a real dataset.

the `rule()` function defines the environment used to generate the data and calculate the correct answers. it is not given to the model during training.

## reproducibility

the random seed is fixed with:

```python
random.seed(42)
```

this means repeated runs produce the same results.

the experiment is also repeated 100 times for each noise level so that the results are not based on a single random train/test split.

## running the experiment

install the dependencies:

```bash
pip install -r requirements.txt
```

run the experiment:

```bash
python main.py
```

the generated graphs are saved in `results/`.

run the tests:

```bash
python -m pytest
```

## project structure

```text
reason/
├── main.py
├── README.md
├── requirements.txt
├── test/
│   └── test_reason.py
└── results/
    ├── noise_results.png
    ├── model_error.png
    └── uncertainty.png
```

## next

the current experiment uses a simple linear relationship, which makes the underlying pattern relatively easy for linear regression to learn.

the next step would be to replace it with a less predictable relationship and repeat the same experiments to see whether the behaviour around noise, distribution shift, and uncertainty changes.
