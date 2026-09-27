from main import make_data, train, get_error


def test_make_data():
    data = make_data([1, 2, 3], noise=0)

    assert data == [
        (1, 2),
        (2, 4),
        (3, 6)
    ]


def test_model_learns_rule():
    data = make_data([1, 2, 3, 4, 5], noise=0)

    model = train(data)

    predictions = model.predict([[6], [7]])

    assert abs(predictions[0] - 12) < 0.001
    assert abs(predictions[1] - 14) < 0.001


def test_error():
    predictions = [10, 20, 30]
    actual = [11, 18, 33]

    error = get_error(predictions, actual)

    assert abs(error - 2) < 0.001