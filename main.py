import random

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


random.seed(42)


def rule(x):
    return x * 2


def make_data(numbers, noise=0):
    data = []

    for number in numbers:
        answer = rule(number)

        if noise > 0:
            answer += random.uniform(-noise, noise)

        data.append((number, answer))

    return data


def train(data):
    x = []
    y = []

    for number, answer in data:
        x.append([number])
        y.append(answer)

    model = LinearRegression()
    model.fit(x, y)

    return model


def get_error(predictions, actual):
    total = 0

    for prediction, answer in zip(predictions, actual):
        total += abs(prediction - answer)

    return total / len(predictions)


def run_test(test_numbers, noise):
    train_numbers = random.sample(range(1, 21), 12)

    train_data = make_data(train_numbers, noise)
    model = train(train_data)

    x = []

    for number in test_numbers:
        x.append([number])

    predictions = model.predict(x)

    actual = []

    for number in test_numbers:
        actual.append(rule(number))

    return predictions, actual


noise_levels = [0, 1, 2, 4, 8]

normal_results = []
shift_results = []

for noise in noise_levels:
    normal_errors = []
    shift_errors = []

    for _ in range(100):

        test_numbers = random.sample(range(1, 21), 8)

        predictions, actual = run_test(
            test_numbers,
            noise
        )

        normal_errors.append(
            get_error(predictions, actual)
        )

        shifted_numbers = list(range(21, 31))

        predictions, actual = run_test(
            shifted_numbers,
            noise
        )

        shift_errors.append(
            get_error(predictions, actual)
        )

    normal_average = sum(normal_errors) / len(normal_errors)
    shift_average = sum(shift_errors) / len(shift_errors)

    normal_results.append(normal_average)
    shift_results.append(shift_average)

    print(
        "noise:",
        noise,
        "| normal:",
        round(normal_average, 3),
        "| shifted:",
        round(shift_average, 3)
    )


numbers = [5, 10, 15, 20, 25]

uncertainty_results = []

for number in numbers:
    predictions = []

    for _ in range(100):
        prediction, actual = run_test(
            [number],
            4
        )

        predictions.append(prediction[0])

    average = sum(predictions) / len(predictions)

    variance = 0

    for prediction in predictions:
        variance += (prediction - average) ** 2

    uncertainty = (variance / len(predictions)) ** 0.5

    actual_error = abs(
        average - rule(number)
    )

    uncertainty_results.append(
        (number, average, uncertainty, actual_error)
    )


print("\nuncertainty")

for number, prediction, uncertainty, actual_error in uncertainty_results:
    print(
        "input:",
        number,
        "| prediction:",
        round(prediction, 2),
        "| uncertainty:",
        round(uncertainty, 2),
        "| actual error:",
        round(actual_error, 2)
    )


plt.figure()

plt.plot(
    noise_levels,
    normal_results,
    marker="o",
    label="normal"
)

plt.plot(
    noise_levels,
    shift_results,
    marker="o",
    label="distribution shift"
)

plt.xlabel("training noise")
plt.ylabel("mean absolute error")
plt.title("model error under noise and distribution shift")
plt.legend()

plt.savefig("model_error.png")
plt.close()


input_numbers = []
uncertainties = []
actual_errors = []

for number, prediction, uncertainty, actual_error in uncertainty_results:
    input_numbers.append(number)
    uncertainties.append(uncertainty)
    actual_errors.append(actual_error)


plt.figure()

plt.plot(
    input_numbers,
    uncertainties,
    marker="o",
    label="uncertainty"
)

plt.plot(
    input_numbers,
    actual_errors,
    marker="o",
    label="actual error"
)

plt.xlabel("input")
plt.ylabel("error")
plt.title("uncertainty vs actual error")
plt.legend()

plt.savefig("uncertainty.png")
plt.close()


print("\nfinished")
print("saved: model_error.png")
print("saved: uncertainty.png")