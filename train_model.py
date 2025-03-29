import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("cleaned_inventory_data.csv")

label_encoder = LabelEncoder()
df['Product ID'] = label_encoder.fit_transform(df['Product ID'].astype(str))


df['Max Units Required'] = df[['Units Sold', 'Units Ordered', 'Demand Forecast']].max(axis=1)
df_grouped = df.groupby(['Year', 'Month', 'Product ID'])['Max Units Required'].max().reset_index()


X = df_grouped[['Year', 'Month', 'Product ID']]
y = df_grouped['Max Units Required']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the model and label encoder
joblib.dump(model, 'model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

def predict_demand(year, month, product_id):
    try:
        product_encoded = label_encoder.transform([str(product_id)])[0] 
        prediction = model.predict([[year, month, product_encoded]])
        return round(prediction[0])
    except ValueError:

        return f"Error: Product ID '{product_id}' is not in the training data."

predicted_units = predict_demand(2025, 4, "P0001")  
print(f"Predicted demand for Product_123 in April 2025: {predicted_units} units")
