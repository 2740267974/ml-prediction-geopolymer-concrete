from sklearn.model_selection import train_test_split
from src.data_process import DataHandler
from src.model_train import train_lightgbm
from src.plotter import DataPlotter

# Data loading
data_handler = DataHandler("data/origin_data.xlsx", n_features=16)
data_handler.load_data()

X, y = data_handler.a, data_handler.b
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=167
)

# Model training
result = train_lightgbm(x_train, y_train, x_test, y_test)

# Plotting
plotter = DataPlotter()
plotter.plot(
    y_train,
    result.y_train_pred,
    y_test,
    result.y_test_pred,
    result.metrics["train"]["r2"],
    result.metrics["test"]["r2"],
    "results/LightGBM.png",
)
