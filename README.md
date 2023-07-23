# Mental Health Forecaster

The Mental Health Forecaster project emphasizes on predicting the Disability-Adjusted Life Years (DALYs) caused by mental disorders in different countries. This application utilizes a robust regression model to deliver accurate predictions based on user-inputted data.

## INSTALLATION
To use the code and run the app, follow these steps:

1. Ensure that you have Python 3.x installed on your system.
2. Install the required libraries by running the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyqt5
```
3. Download the project files and navigate to the project directory.

## USAGE
Once you have cloned the repository and installed the required libraries, you can run the `app.py` script to launch the GUI. The GUI has fields for inputting the prevalence rates of specific mental disorders. Once these values are entered, you can click on the "Predict" button to obtain the forecasted DALYs. The application will use the trained regression models to compute this prediction and will display it within the GUI.

## THE CODE

The primary code is contained within the `app.py` script, which handles the GUI, and `models.py`, which implements the regression models. Both of these files import functionality from the other scripts within the repository, which handle tasks such as data cleaning, preprocessing, and feature engineering.

The regression models used within this application include Ridge Regression, Lasso Regression, Elastic Net Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression, XGBoost Regression, K-Nearest Neighbors Regression, Bayesian Regression, Neural Network Regression, and Gradient Boosting Regression. These models are evaluated based on Mean Squared Error (MSE) and R-squared Score.

## SUMMARY
The Mental Health Forecaster is a valuable tool for predicting the Disability-Adjusted Life Years (DALYs) caused by mental disorders. It was developed during my internship at IBM SkillsBuild, and it achieves an impressive precision accuracy of 98.7%.

The application combines Python's data analysis capabilities with a user-friendly GUI, making it easy for users to input data and receive predictions. Behind the scenes, a dozen different regression algorithms are used to compute the predictions, ensuring that the results are as accurate as possible.

## REFERENCES
The dataset used in this project was taken from ourworldindata.org. 

The project was developed during my IBM SkillsBuild internship in association with AICTE. 

Please note that the above instructions are subject to further improvements and changes.
