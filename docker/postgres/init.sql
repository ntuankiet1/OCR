-- One Postgres instance, one database per service.
CREATE DATABASE mlflow;
CREATE DATABASE labelstudio;
-- Feedback store: review-queue exports land here before `parcel-ocr ingest`.
CREATE DATABASE feedback;
