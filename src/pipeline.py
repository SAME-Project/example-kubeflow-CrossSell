from kfp import components
from kfp import dsl

from kubernetes.client.models import V1EnvVar

@dsl.pipeline(
    name="Cross-selling training pipeline",
    description="",
)
def pipeline(
    # Lots of epochs highlights the value of a streaming metrics system like
    # mlflow
    epochs: int = 1000,
    # TODO find a less ugly way to sneak these in (same CLI can do this for us?
    # WITH METAPROGRAMMING?) - or, actually, just using these defaults are
    # probably fine
    AWS_ACCESS_KEY_ID: str = "minio",
    AWS_SECRET_ACCESS_KEY: str = "minio123",
    MLFLOW_S3_ENDPOINT_URL: str = "http://minio.mlflow.svc.cluster.local:9000",
    MLFLOW_TRACKING_URI: str = "http://mlflow.mlflow.svc.cluster.local:5000",
):
    import download
    import metrics
    import preprocessing
    import training
    import util

    BASE_IMAGE = "gcr.io/deeplearning-platform-release/tf2-cpu.2-4:m65"

    downloadOp = components.func_to_container_op(
        download.extract_tar_from_url, base_image=BASE_IMAGE
    )()

    preprocessOp = components.func_to_container_op(
        preprocessing.preprocess, base_image=BASE_IMAGE
    )(downloadOp.output)


    trainOp = components.func_to_container_op(
        training.train, base_image=BASE_IMAGE,
        packages_to_install=["mlflow==1.17.0", "boto3==1.17.79"],
    )(
        preprocessOp.output,
        n_epochs=epochs,
    )
    for (k, v) in [("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
                   ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
                   ("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL),
                   ("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)]:
        trainOp = trainOp.add_env_variable(V1EnvVar(name=k, value=v))

    components.func_to_container_op(metrics.produce_metrics, base_image=BASE_IMAGE)(
        preprocessOp.output, trainOp.output
    )

    components.func_to_container_op(util.generate_submission, base_image=BASE_IMAGE)(
        preprocessOp.output, trainOp.output
    )
