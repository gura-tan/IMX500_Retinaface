import tensorflow as tf
from retinaface import RetinaFace
import os

EXPORT_DIR = "retinaface_savedmodel"

class RetinaFaceModule(tf.Module):
    def __init__(self):
        super().__init__()
        # RetinaFace 本体を「属性」として保持するのが超重要
        self.model = RetinaFace.build_model()

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, None, 3],
                dtype=tf.float32,
                name="input_image"
            )
        ]
    )
    def __call__(self, x):
        return self.model(x)


def main():
    module = RetinaFaceModule()

    # 一度だけ forward して変数を確定させる
    dummy_input = tf.random.uniform(
        shape=(1, 640, 640, 3),
        dtype=tf.float32
    )
    _ = module(dummy_input)

    # SavedModel export
    tf.saved_model.save(
        module,
        EXPORT_DIR,
        signatures=module.__call__
    )

    print(f"SavedModel exported to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
