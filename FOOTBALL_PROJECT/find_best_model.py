import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from prepare_data import prepare_data


def find_best_model(league_name, start=0.001, end=1, jump=0.001):
    x_train, y_train, x_val, y_val, x_test, y_test, scaler, team_columns = prepare_data(league_name)

    scores = []
    for c in np.arange(start, end + jump, jump):
        model = LogisticRegression(C=c)
        model.fit(x_train, y_train)
        score = model.score(x_val, y_val)
        scores.append(score)

    best_index = np.argmax(scores)
    best_c = round((best_index + start / jump) * jump, 5)

    print(f"Best C: {best_c}, Accuracy: {scores[best_index]}\n")

    model = LogisticRegression(C=best_c)
    model.fit(x_train, y_train)

    print("******************************************** - Validation - ********************************************")
    print(classification_report(y_val, model.predict(x_val)))
    print(y_val.value_counts() / len(y_val))

    print("*********************************************** - Test - ***********************************************")
    print(classification_report(y_test, model.predict(x_test)))
    print(y_test.value_counts() / len(y_test))

    return model, scaler, team_columns
