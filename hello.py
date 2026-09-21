import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = pd.read_csv('Maths.csv')

#print(df.info())
#print(df.dtypes) 
numerical_features =[]
categorical_features =[]
for col in df.columns:
    if df[col].dtype == 'str':
        categorical_features.append(col)
    else:
        numerical_features.append(col)
#print("Numerical Features:", numerical_features)
#print("Categorical Features:", categorical_features)
'''for col in categorical_features:
    print("\n", col)
    print(df[col].unique())
for col in numerical_features:
    print("\n", col)
    print(df[col].unique())'''
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_features])
print("Encoded Data Shape:", encoded_data.shape)
x = df.drop(columns='G3')
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features[:-1]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("Train shape:", X_train_encoded.shape)
print("Test shape:", X_test_encoded.shape)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("r2 train Score:", r2_score(y_train, model.predict(X_train)))
print ("R2 Score:", r2_score(y_test, y_pred))
score = cross_val_score(model, x, y, cv=5, scoring='r2')
print("Cross-Validation R2 Scores:", score)
print("Mean Cross-Validation R2 Score:", np.mean(score))
print("Std Dev Cross-Validation R2 Score:", np.std(score))

'''feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importances = model.named_steps["regressor"].feature_importances_

for name, importance in sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
):
    print(name, importance)'''
result = permutation_importance(model, X_test, y_test, scoring='r2', n_repeats=10, random_state=42)
feature_names = X_test.columns

importance = result.importances_mean

pairs = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)

for name, value in pairs[:10]:
    print(name, value)
'''plt.scatter(df['G1'], df['G2'])
plt.xlabel('G1')
plt.ylabel('G2')
plt.title('Relationship between G1 and G2')
plt.show()'''

'''print ("G1 and G3:", df[['G1', 'G3']].corr())
print ("G2 and G3:", df[['G2', 'G3']].corr())'''

#x = df[['G1', 'G2']]
#y = df['G3']

'''x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
}
for name, model in models.items():
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    print(f"{name} R2 Score: {np.mean(scores)}")'''
'''model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)'''

'''print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)'''

'''train_r2 = r2_score(y_train, model.predict(x_train))
test_r2 = r2_score(y_test, y_pred)'''


