import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create the main application window
app = tk.Tk()
app.title("AI Mental Health Forecaster")
app.geometry("800x600")

# Disable fullscreen
app.resizable(False, False)

# Set the app icon
app.iconphoto(True, tk.PhotoImage(file="app_icon.png"))

# Create the left panel
left_panel = tk.Frame(app, bg="#f2f2f2", width=200*app.winfo_screenwidth(), height=app.winfo_screenheight())
left_panel.pack(side="left", fill="y")

# Add the image to the left panel and resize it
image = tk.PhotoImage(file="cover2.png")  # Modify the subsample values as needed
img_label = tk.Label(left_panel, image=image)
img_label.pack(fill="y", padx=0, pady=0, expand=True)

# Create the right panel
right_panel = tk.Frame(app, bg="white")
right_panel.pack(side="right", fill="both", expand=True)

# Add the heading to the right panel
heading_label = tk.Label(right_panel, text="AI MENTAL HEALTH FORECASTER", font=("Roboto Mono", 16))
heading_label.pack(pady=20)

# Create the form
form = ttk.Frame(right_panel)
form.pack()

# Create the input fields
input_fields = [
    {"label": "Prevalence - Schizophrenia:", "id": "schizophrenia"},
    {"label": "Prevalence - Bipolar disorder:", "id": "bipolar"},
    {"label": "Prevalence - Eating disorders:", "id": "eating_disorders"},
    {"label": "Prevalence - Anxiety disorders:", "id": "anxiety_disorders"},
    {"label": "Prevalence - Drug use disorders:", "id": "drug_use_disorders"},
    {"label": "Prevalence - Depressive disorders:", "id": "depressive_disorders"},
    {"label": "Prevalence - Alcohol use disorders:", "id": "alcohol_use_disorders"},
]

input_entries = {}

for field in input_fields:
    input_frame = ttk.Frame(form)
    input_frame.pack(pady=10)

    label = ttk.Label(input_frame, text=field["label"])
    label.pack(side="left")

    input_entry = ttk.Entry(input_frame, width=20)
    input_entry.pack(side="left")
    input_entries[field["id"]] = input_entry

# Add the ML model selection
ml_model_frame = ttk.Frame(form)
ml_model_frame.pack(pady=10)

ml_model_label = ttk.Label(ml_model_frame, text="Select ML Model:")
ml_model_label.pack(side="left", padx=5)

ml_model_combobox = ttk.Combobox(ml_model_frame, values=[
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net Regression",
    "Linear Regression",
    "Bayesian Regression",
    "SVR",
    "Decision Tree Regression",
    "Random Forest Regression",
    "XGBoost Regression",
    "K-Nearest Neighbors Regression",
    "MLP Regression",
    "Gradient Boosting Regression"
])
ml_model_combobox.pack(side="left")

# Add the predict button
predict_button = ttk.Button(form, text="PREDICT")
predict_button.pack(pady=20)

# Add some space below the Predicted Result Value
result_space_label = ttk.Label(form, text="", font=("Roboto Mono", 10))
result_space_label.pack()

# Add the result box
result_frame = ttk.Frame(right_panel)
result_frame.pack()

result_label = ttk.Label(result_frame, text="", font=("Roboto Mono", 16), background="white", borderwidth=2, relief="solid", wraplength=600, anchor="center")
result_label.pack(pady=10, padx=20)

# Add the footer
footer_frame = ttk.Frame(right_panel)
footer_frame.pack(pady=10)

made_by_label = ttk.Label(footer_frame, text="Made by Harsh Raj Singh Rathore")
made_by_label.pack()

github_label = ttk.Label(footer_frame, text="Github: @heyhrsh", foreground="#1a5dd8", cursor="hand2")
github_label.pack()

# Configure the GitHub label to open the URL in a web browser
def open_github(event):
    import webbrowser
    webbrowser.open("https://github.com/heyhrsh")

github_label.bind("<Button-1>", open_github)

# Reading CSV Data into Pandas DataFrames
df1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
df2 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')

# Filling Missing Values
numeric_columns = df1.select_dtypes(include=[np.number]).columns
df1[numeric_columns] = df1[numeric_columns].fillna(df1[numeric_columns].mean())

numeric_columns = df2.select_dtypes(include=[np.number]).columns
df2[numeric_columns] = df2[numeric_columns].fillna(df2[numeric_columns].mean())

merged_df = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'])

X = merged_df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)']]

y = merged_df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and their names
models = {
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net Regression': ElasticNet(),
    'Linear Regression': LinearRegression(),
    'Bayesian Regression': BayesianRidge(),
    'SVR': SVR(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'XGBoost Regression': XGBRegressor(),
    'K-Nearest Neighbors Regression': KNeighborsRegressor(),
    'MLP Regression': MLPRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor()
}

# Create the predict function
def predict():
    # Get user inputs
    input_data = [float(input_entries[field_id].get()) for field_id in input_entries]
    selected_model = ml_model_combobox.get()

    # Check if all input fields are filled
    if not all(input_data):
        result_label.config(text="Please fill all input fields.", foreground="red")
        return

    # Preprocess user inputs (if required)
    input_data = np.array(input_data).reshape(1, -1)

    # Create the selected model instance and make predictions
    model = models[selected_model]

    # Fit the model using the training data
    model.fit(X_train, y_train)

    # Make predictions
    prediction = model.predict(input_data)

    # Display the prediction result on the app window
    result_label.config(text=f"Predicted DALYs: {prediction[0]:.2f}", foreground="black")

# Add the predict button click event
predict_button.config(command=predict)

# Start the main event loop
app.mainloop()
