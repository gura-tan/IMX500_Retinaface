#quantize_retinaface.py
import os
import cv2
import numpy as np
import tensorflow as tf
import model_compression_toolkit as mct

IMG_SIZE = 320
BATCH_SIZE = 1

# -------------------------
# キャリブレーション画像ローダ(フォルダから1枚ずつ画像を読み出し)
# -------------------------
def calibration_dataset_from_folder(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png"))
    ]

    def generator():
        for path in image_files:
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32)

            # RetinaFace は [0,255] 前提
            img = np.expand_dims(img, axis=0)
            yield [img]

    return generator


if __name__ == "__main__":
    CALIB_DIR = "calib_images"

    model = tf.keras.models.load_model(
        "models/retinaface_savedmodel"
    )

    repr_dataset = calibration_dataset_from_folder(CALIB_DIR)

    quantized_model, _ = mct.ptq.keras_post_training_quantization(
        model,
        representative_data_gen=repr_dataset,
        n_iter=50,   # calibration step
    )#TPCが明示されていませんが、DEFAULT_KERAS_TPCがimx500を指定(おそらく)

    quantized_model.save(
        "models/retinaface_quantized",
        include_optimizer=False,
        save_format="tf"
    )

    print("Quantized model saved.")
	