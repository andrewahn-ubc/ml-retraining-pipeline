from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

default_args = {
    'owner': 'andrew',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "stock model retraining",
    default_args = default_args,
    description = "Automated stock prediction model retraining pipeline",
    schedule_interval = '0 0 * * *', # format: 'minute    hour    day-of-month    month    day-of-week'
    start_date=datetime(2025,1,1), # from when our dag should start running (if before current date and catchup=false, then we start tmr)
    catchup=False,
    tags=['stocks', 'ml', 'production']
)

def load_data(**context):
    ticker="SPY"
    data = yf.download(ticker, period="2y", progress=False)
    if data.empty:
        raise ValueError("Failed to fetch stock data")
    data_path = Path("/opt/airflow/data")
    data_path.mkdir(exist_ok=True)
    data.to_csv(data_path / "raw_stock_data.csv")
    print("successfully fetched data")

def engineer_features(**content):
    data_path = Path("/opt/airflow/data")
    df = pd.read_csv(data_path / "raw_stock_data.csv", index_col=0, parse_dates=True)

    # create features
    df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["price_change"] = df['Close'].pct_change() * 100
    df["volume_change"] = df["Volume"].pct_change() * 100

    # create target variable
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()

    # only keep the relevant columns
    features = ["SMA_10", "SMA_50", "RSI", "price_change", "volume_change"]
    X = df[features].to_numpy()
    y = df["target"].to_numpy()

    # split data
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)    

    # save data
    np.save(data_path / "X_train.npy", X_train)
    np.save(data_path / "X_test.npy", X_test)
    np.save(data_path / "y_train.npy", y_train)
    np.save(data_path / "y_test.npy", y_test)

    print("successfully engineered features")
    return {"train_size": len(X_train), "test_size": len(X_test)}

def train_model(**context):
    data_path = Path("/opt/airflow/data")
    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")

    # standardize features
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)

    # train model
    model = LogisticRegression(max_iter=1000, random_state=67, class_weight="balanced")
    model.fit(X_train_standardized, y_train)

    # save model and scaler
    model_path = Path("/opt/airflow/models")
    model_path.mkdir(exist_ok=True)
    model_package = {
        "model": model,
        "scaler": scaler
    }
    with open(model_path / "model_new.pkl", "wb") as f:
        pickle.dump(model_package, f)

    print("finished training and saving model")
    return {"status": "trained"}

def validate_model(**context):
    pass

def deploy_model(**context):
    pass

task_load = PythonOperator(
    task_id = "load_data",
    python_callable=load_data,
    dag=dag
)

task_features = PythonOperator(
    task_id = "engineer_features",
    python_callable=engineer_features,
    dag=dag
)

task_train = PythonOperator(
    task_id = "train_model",
    python_callable=train_model,
    dag=dag
)

task_validate = PythonOperator(
    task_id = "validate_model",
    python_callable=validate_model,
    dag=dag
)

task_deploy = PythonOperator(
    task_id = "deploy_model",
    python_callable=deploy_model,
    dag=dag
)

task_notify = EmailOperator(
    task_id = "send_email",
    to="andrewahn21@gmail.com",
    subject="Stock Prediction Model Training Complete",
    html_content="Test",
    dag=dag
)

task_load >> task_features >> task_train >> task_validate >> task_deploy >> task_notify