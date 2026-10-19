# MLflow tracking server + model registry.
# The upstream image ships without the Postgres driver and boto3 (S3 artifact store).
FROM ghcr.io/mlflow/mlflow:v3.17.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.* boto3==1.43.*
