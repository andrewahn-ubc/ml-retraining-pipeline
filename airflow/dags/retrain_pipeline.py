from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta

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
    pass

def train_model(**context):
    pass

def validate_model(**context):
    pass

def deploy_model(**context):
    pass

task_load = PythonOperator(
    task_id = "load_data",
    python_callable=load_data,
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

task_load >> task_train >> task_validate >> task_deploy >> task_notify