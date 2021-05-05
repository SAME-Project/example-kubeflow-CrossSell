from kfp import components
from kfp import dsl


@dsl.pipeline(
    name="Cross-selling training pipeline",
    description="",
)
def pipeline(
    epochs: int = 10,
):
    from src import download
    from src import metrics
    from src import preprocessing
    from src import training
    from src import util

    BASE_IMAGE = "gcr.io/deeplearning-platform-release/tf2-cpu.2-4:m65"

    downloadOp = components.func_to_container_op(
        download.extract_tar_from_url, base_image=BASE_IMAGE
    )()

    preprocessOp = components.func_to_container_op(
        preprocessing.preprocess, base_image=BASE_IMAGE
    )(downloadOp.output)

    trainOp = components.func_to_container_op(training.train, base_image=BASE_IMAGE)(
        preprocessOp.output,
        n_epochs=epochs,
    )

    components.func_to_container_op(metrics.produce_metrics, base_image=BASE_IMAGE)(
        preprocessOp.output, trainOp.output
    )

    components.func_to_container_op(util.generate_submission, base_image=BASE_IMAGE)(
        preprocessOp.output, trainOp.output
    )
