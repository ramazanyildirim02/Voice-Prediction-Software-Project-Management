import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def scores():
    model = joblib.load("scores.sav")
    df = pd.read_excel("dataset.xlsx")
    

    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=0)

    y_pred = model.predict(x_test)
    

    acc = accuracy_score(y_test, y_pred)
    fm = f1_score(y_test, y_pred,average="macro")

    # Sonuçları yazdır
    print("Accuracy Skoru:", acc)
    print("F1 Skoru:", fm)