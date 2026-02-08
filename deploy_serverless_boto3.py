import time
import boto3

ROLE_ARN = "arn:aws:iam::329048224619:role/service-role/SageMaker-ML-Engineer"

MODEL_DATA = "s3://news-source-quality-predictor/model.tar.gz"
SOURCE_CODE = "s3://news-source-quality-predictor/source/source.tar.gz"

# Official SageMaker sklearn inference container (CPU)
IMAGE_URI = "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

MEMORY_MB = 3072
MAX_CONCURRENCY = 5

ENABLE_DATA_CAPTURE = False
DATA_CAPTURE_S3 = "s3://news-source-quality-predictor/data-capture/"


def main():
    session = boto3.Session()
    region = session.region_name or "eu-north-1"


    sm = session.client("sagemaker", region_name=region)

    ts = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"nsq-model-{ts}"
    config_name = f"nsq-config-{ts}"
    endpoint_name = f"nsq-endpoint-{ts}"

    # ---- Create Model ----
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": MODEL_DATA,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": SOURCE_CODE,
            },
        },
    )

    # ---- Endpoint Config (Serverless) ----
    config = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": MEMORY_MB,
                    "MaxConcurrency": MAX_CONCURRENCY,
                },
            }
        ],
    }

    # if ENABLE_DATA_CAPTURE:
    #     config["DataCaptureConfig"] = {
    #         "EnableCapture": True,
    #         "InitialSamplingPercentage": 25,
    #         "DestinationS3Uri": DATA_CAPTURE_S3,
    #         "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
    #         "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
    #     }

    sm.create_endpoint_config(**config)

    # ---- Create Endpoint ----
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    print("\n✅ Endpoint creation started")
    print("Endpoint:", endpoint_name)
    print("Region:", region)
    print("\nOpen AWS website → SageMaker → Endpoints → wait for InService")


if __name__ == "__main__":
    main()
