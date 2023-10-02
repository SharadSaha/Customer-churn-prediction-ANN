# 📊 Retainify

Welcome to the Customer Churn Prediction Web Application repository! This application predicts the probability of customer churn based on various inputs like tenure, senior citizen status, multiple lines, internet service, tech support, and more. It has been trained on the Telco customer churn dataset.

## 🚀 What the App Does

This web application is designed to predict the probability of customer churn for businesses and service providers. It takes into account a range of customer-related factors, including:

- **Tenure** 🕒: How long the customer has been with the service provider.
- **Senior Citizen Status** 👴👵: Whether the customer is a senior citizen or not.
- **Multiple Lines** 📞: Whether the customer has multiple phone lines.
- **Internet Service** 🌐: The type of internet service subscribed to.
- **Tech Support** 🛠️: Whether the customer has technical support.
- ... and several other relevant features.

The application leverages a machine learning model trained on historical data to provide an estimate of the likelihood that a customer will churn in the near future. This predictive capability can help businesses take proactive measures to retain valuable customers and reduce churn.

## ℹ️ About the Dataset

The dataset includes information about:

- Customers who left within the last month – the column is called Churn.
- Services that each customer has signed up for, such as phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account information, including how long they've been a customer, contract type, payment method, paperless billing, monthly charges, and total charges.
- Demographic information about customers, including gender, age range, and if they have partners and dependents.

## 🛠️ Technologies Used

- **Frameworks** 🧰: TensorFlow, scikit-learn
- **Web Application** 🌐: Streamlit
- **Language** 🐍: Python
- **Tools** 🔧: Pandas, NumPy, Matplotlib, Seaborn (for data visualization, data cleaning, and feature engineering)

## 🌐 Web Application

Access the Customer Churn Prediction Web Application [here](https://share.streamlit.io/sharadsaha/customer-churn-prediction-ann/main/app.py).

## 🌍 Real-Life Use Cases

Predicting customer churn has valuable applications in various industries, including:

- **Telecommunications** 📞: Identify customers at risk of leaving the service and tailor retention strategies.
- **E-commerce** 🛍️: Improve customer retention and engagement by identifying potential churners.
- **Subscription Services** 💳: Reduce subscription cancellations by targeting at-risk customers with promotions.
- **Finance** 🏦: Predict and mitigate customer attrition for banking and financial services.

## 📊 Exploratory Data Analysis

Exploratory Data Analysis (EDA) has been performed on the dataset to gain insights and inform the modeling process. Visualizations and statistical summaries are available in the project.

## 🧠 Model Selection

Several machine learning algorithms were considered for this project. After thorough experimentation, we chose to implement an Artificial Neural Network (ANN) model for its ability to capture complex relationships in the data and achieve high predictive accuracy.

## 📁 Project Structure

- `app.py`: The main Streamlit application.
- `final_model/`: Contains the pre-trained ANN model.
- `data/`: Data files used for training and testing.
- `notebooks/`: Jupyter notebooks for EDA and model development.
- `requirements.txt`: List of Python dependencies.

## 🚀 How to Run the Project Locally

To run this project locally, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/SharadSaha/Customer-churn-prediction-ANN.git

2. Navigate to the project directory:

   ```bash
    cd your-repo

3. Install the required dependencies

   ```bash
    pip install -r requirements.txt

4. Run the web application using Streamlit

   ```bash
    streamlit run app.py


## 📬 Contact

If you have any questions or feedback, please feel free to reach out:

- [Sharad Saha](mailto:sahasharad29@gmail.com)

We hope this application proves valuable in predicting and mitigating customer churn! 🚀
