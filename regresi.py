import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data from your image
data = {
    'Year': [2012, 2013, 2014, 2015, 2016],
    'Large Mask (units)': [631782, 678927, 638691, 659849, 709362],
    'Large Mask (rupiah)': [11372076000, 12220686000, 11496438000, 11877282000, 12768516000],
    'Small Mask (units)': [223542, 278327, 262891, 265782, 292539],
    'Small Mask (rupiah)': [3800214000, 4731559000, 4469147000, 4518294000, 4973163000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y) - we'll predict 'Large Mask (units)'
X = df[['Year']]  # Year is the feature
y = df['Small Mask (units)']  # Target variable to predict

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model_linreg = LinearRegression()
model_linreg.fit(X_train, y_train)

# Example of predicting couple of years after available data
future_years = pd.DataFrame({'Year': [2017, 2018, 2019]})  # Tahun yang ingin diprediksi

# future predict
y_pred_future = model_linreg.predict(future_years)

# Menampilkan hasil prediksi untuk tahun mendatang
print(list(zip(future_years['Year'], y_pred_future)))

