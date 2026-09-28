import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Data Set/Car Price Prediction.csv")

print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())

features = ["enginesize", "horsepower", "curbweight", "citympg"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression R2 Score:", r2_lr)

X_poly_feature = df[["horsepower"]].values
y_poly_target = df["price"].values

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly_feature, y_poly_target, test_size=0.2, random_state=42)

degrees = [2, 3, 4]
results = {}

plt.figure(figsize=(9, 6))
plt.scatter(X_poly_feature, y_poly_target, color="blue", label="Actual Data", alpha=0.5)

x_line = np.linspace(X_poly_feature.min(), X_poly_feature.max(), 200).reshape(-1, 1)
colors = ["red", "green", "orange"]

for degree, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=degree)
    Xp_train_poly = poly.fit_transform(Xp_train)
    Xp_test_poly = poly.transform(Xp_test)

    poly_model = LinearRegression()
    poly_model.fit(Xp_train_poly, yp_train)

    yp_pred = poly_model.predict(Xp_test_poly)

    rmse_poly = np.sqrt(mean_squared_error(yp_test, yp_pred))
    r2_poly = r2_score(yp_test, yp_pred)

    results[degree] = {"RMSE": rmse_poly, "R2": r2_poly}

    y_line = poly_model.predict(poly.transform(x_line))
    plt.plot(x_line, y_line, color=color, linewidth=2, label=f"Degree {degree}")

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Polynomial Regression - Fitted Curves")
plt.legend()
plt.show()

for degree in degrees:
    print("Degree:", degree, "RMSE:", results[degree]["RMSE"], "R2:", results[degree]["R2"])

print("Linear Regression RMSE:", rmse_lr, "R2:", r2_lr)
