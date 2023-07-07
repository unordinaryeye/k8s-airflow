from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


def preprocess_data():
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    

def train_model():
    x_train_data = np.load('x_train.npy')
    y_train_data = np.load('y_train.npy')

    model = SGDRegressor(verbose=1)
    model.fit(x_train_data, y_train_data)

    joblib.dump(model, 'model.pkl')


def test_model():
    x_test_data = np.load('x_test.npy')
    y_test_data = np.load('y_test.npy')

    model = joblib.load('model.pkl')
    y_pred = model.predict(x_test_data)

    err = mean_squared_error(y_test_data, y_pred)

    with open('output.txt', 'a') as f:
        f.write(str(err))


def deploy_model():
    print('deploying model...')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 7, 5),
)

t1 = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id='test_model',
    python_callable=test_model,
    dag=dag,
)

t4 = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

t1 >> t2 >> t3 >> t4