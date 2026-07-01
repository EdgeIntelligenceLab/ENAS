"""
INT8 Post-Training Quantization utilities for ENAS v2.1.

Handles TFLite conversion with custom temp directory (avoids /tmp overflow
on shared servers) and provides representative dataset wrapping.
"""

import os
import tempfile
import shutil
import tensorflow as tf


def quantize_model(model, train_ds, output_path, custom_tmp_dir=None,
                   num_samples=150):
    """
    Convert a Keras model to INT8 TFLite with post-training quantization.

    Parameters
    ----------
    model : tf.keras.Model
        Trained float model.
    train_ds : tf.data.Dataset
        Training dataset (used for representative samples).
    output_path : str or Path
        Where to save the .tflite file.
    custom_tmp_dir : str or None
        Custom directory for TFLite's intermediate files (avoids /tmp).
    num_samples : int
        Number of representative samples for calibration.

    Returns
    -------
    Path
        Path to the saved .tflite file.
    """
    def representative_dataset():
        for data in train_ds.rebatch(1).take(num_samples):
            yield [tf.dtypes.cast(data[0], tf.float32)]

    old_tempdir = tempfile.tempdir
    if custom_tmp_dir is not None:
        os.makedirs(custom_tmp_dir, exist_ok=True)
        tempfile.tempdir = custom_tmp_dir

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset    = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
    finally:
        tempfile.tempdir = old_tempdir
        if custom_tmp_dir is not None:
            shutil.rmtree(custom_tmp_dir, ignore_errors=True)

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def evaluate_tflite(tflite_path, test_ds):
    """
    Evaluate an INT8 TFLite model on a test dataset.

    Parameters
    ----------
    tflite_path : str or Path
    test_ds : tf.data.Dataset

    Returns
    -------
    float
        Test accuracy in [0, 1].
    """
    interpreter = tf.lite.Interpreter(str(tflite_path))
    interpreter.allocate_tensors()
    out_det = interpreter.get_output_details()[0]
    in_det  = interpreter.get_input_details()[0]

    correct = wrong = 0
    for image, label in test_ds.rebatch(1):
        if in_det['dtype'] == tf.uint8:
            scale, zp = in_det["quantization"]
            image = image / scale + zp
        interpreter.set_tensor(
            in_det['index'], tf.dtypes.cast(image, tf.uint8)
        )
        interpreter.invoke()
        if label.numpy().argmax() == interpreter.get_tensor(
                out_det['index']).argmax():
            correct += 1
        else:
            wrong += 1

    return correct / (correct + wrong)
