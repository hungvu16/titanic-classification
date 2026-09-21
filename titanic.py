import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
df = pd.read_csv('Titanic-Dataset.csv')
'''print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df['Survived'].value_counts())'''
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate PassengerId:", df["PassengerId"].duplicated().sum())
X = df.drop(columns=[
    "Survived",
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
])

y = df["Survived"]
numerical_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked"
]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessed training set shape:", X_train_processed.shape)
print("Preprocessed test set shape:", X_test_processed.shape)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100, random_state=42,max_depth=3, learning_rate=0.1,eval_metric='logloss'))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("CV Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
print("Std:", scores.std())
'''from sklearn.metrics import accuracy_score

for depth in [1, 2, 3, 5, 10, 20]:
    model_dt = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    model_dt.fit(X_train, y_train)

    train_pred = model_dt.predict(X_train)
    test_pred = model_dt.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(
        f"Depth = {depth}: "
        f"Train Accuracy = {train_acc:.4f}, "
        f"Test Accuracy = {test_acc:.4f}"
    )
    scores = cross_val_score(
        model_dt,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"Depth = {depth}")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())
    print()'''