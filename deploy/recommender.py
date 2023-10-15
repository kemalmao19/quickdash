import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
predictData = lambda model, data: crop_names.get(model.predict(data)[0])

warnings.filterwarnings("ignore")

path_data = "/Users/kemalmao/crop_recommender_ML/dataset/Crop_recommendation.csv"

# current_data = pd.read_csv(input("input the data path: "))

X, y = xySplitter(readFile(path_data))
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = trainTest(X, y)

trained_model = modeling(LogisticRegression(), X_train, y_train)

y_pred = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualization
def nutrient():
    # Create a pie chart to visualize the accuracy
    labels = ['Correct Predictions', 'Incorrect Predictions']
    sizes = [accuracy * len(y_test), (1 - accuracy) * len(y_test)]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode 1st slice (i.e., 'Correct Predictions')

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.7, 'edgecolor': 'w'})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.gcf().set_facecolor('none') #transparent

    # Save the donut chart to a BytesIO object
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Encode the image in base64 to embed it in HTML
    img_base64 = base64.b64encode(img_io.read()).decode()
    
    # Close the plot to release resources
    plt.close()
    
    return img_base64
# print("=" * 60)
# print(current_data)
# predictData(trained_model, current_data)
