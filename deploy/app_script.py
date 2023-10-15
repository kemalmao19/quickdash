import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Preprocessing
readFile = lambda path: pd.read_csv(path)

def xySplitter(data):
    cols = data.columns
    X = data[cols[:-1]]
    y = data[cols[-1]]
    return X, y

# Training
trainTest = lambda x, y: train_test_split(x, y, test_size=0.3, random_state=42)

modeling = lambda model, X_train, y_train: model.fit(X_train, y_train)

crop_names = {
    13: "rice",
    9: "maize",
    0: "soybeans",
    3: "beans",
    12: "peas",
    8: "groundnuts",
    6: "cowpeas",
    2: "banana",
    10: "mango",
    7: "grapes",
    14: "watermelon",
    1: "apple",
    11: "orange",
    5: "cotton",
    4: "coffee",
}

# Predicting
predictData = lambda model, data: print(
    "=" * 60,
    f'Crop Recommendation: {crop_names.get(model.predict(data)[0], "Unknown")}',
    "=" * 60,
)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path_data = "/Users/kemalmao/crop_recommender_ML/dataset/Crop_recommendation.csv"

    current_data = pd.read_csv(
        input("input the data path: ")
    )  

    X, y = xySplitter(readFile(path_data))
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = trainTest(X, y)

    trained_model = modeling(LogisticRegression(), X_train, y_train)

    print("=" * 60)
    print(current_data)
    predictData(trained_model, current_data)
