import pandas as pd
import numpy as np
from regression import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score


def linear():
    df = pd.read_excel("Lab1/data/CCPP.xlsx")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    s = np.insert(reg.coef_.T, 0, reg.intercept_)

    print("Scikit-Learn")
    print(np.vstack(s))
    print("MAE:", mean_absolute_error(y_train, reg.predict(X_train)))
    print("R2:")
    print("Train:", r2_score(y_train, reg.predict(X_train)))
    print("Test:", r2_score(y_test, reg.predict(X_test)))

    print("\nRegression")
    model = LinearRegression(df)
    model.gradientDescent(alpha=1, iter=1e4, threshold=1e-6)
    print(model.w)
    print("MAE:", model.MAE)
    print("R2:")
    print("Train:", model.R2)
    print("Test:", model.R2_test)
    model.plotCost()


def logistic():
    df2 = pd.read_csv("Lab3/data/loan_prediction.csv")
    df2.Gender.fillna("Male", inplace=True)
    df2.dropna(subset=["Married"], inplace=True)
    df2.Dependents.fillna("0", inplace=True)
    df2.Self_Employed.fillna("No", inplace=True)
    df2.Credit_History.fillna(df2.Loan_Status, inplace=True)
    df2.Credit_History.replace(["Y", "N"], [1, 0], inplace=True)
    df2.LoanAmount.fillna(df2.LoanAmount.median(), inplace=True)
    df2.Loan_Amount_Term.fillna(df2.Loan_Amount_Term.mode()[0], inplace=True)
    df2.drop("Loan_ID", axis=1, inplace=True)
    df2.replace(["Female", "Male"], [0, 1], inplace=True)
    df2.replace(["No", "Yes"], [0, 1], inplace=True)
    df2.replace(["Not Graduate", "Graduate"], [0, 1], inplace=True)
    df2.replace(["0", "1", "2", "3+"], [0, 1, 2, 3], inplace=True)
    df2.replace(["Rural", "Semiurban", "Urban"], [0, 1, 2], inplace=True)
    df2.replace(["N", "Y"], [0, 1], inplace=True)

    X = df2.iloc[:, :-1]
    y = df2.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = linear_model.LogisticRegression()
    reg.fit(X_train, y_train)
    s = np.insert(reg.coef_.T, 0, reg.intercept_)

    print("Scikit-Learn")
    print(np.vstack(s))
    print("Score:", reg.score(X_train, y_train))
    print("Test score:", reg.score(X_test, y_test))
    print("Precision:", precision_score(y_train, reg.predict(X_train)))
    print("Recall:", recall_score(y_train, reg.predict(X_train)))
    print("f1:", f1_score(y_train, reg.predict(X_train)))

    print("\nRegression")
    model = LogisticRegression(df2, random_state=0)
    model.gradientDescent(alpha=0.003, threshold=1e-9, iter=1e4)
    print(model.w)
    print("Matrix:\n", model.matrix)
    print("Accuracy:", model.acc)
    print("Accuracy test:", model.acc_test)
    print("Precision:", model.pre)
    print("Precision test:", model.pre_test)
    print("Recall:", model.rec)
    print("f1:", model.f1)
    model.plotCost()


linear()
logistic()
