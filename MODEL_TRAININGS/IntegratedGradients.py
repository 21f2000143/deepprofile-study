################################################################
# Integrated Gradients (TF2 / Keras version)
# Original author: Naozumi Hiranuma
# Updated for TensorFlow 2.x
################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential


class IntegratedGradients:
    def __init__(self, model, outchannels=None, verbose=1):
        """
        model: tf.keras Model you want to explain
        outchannels: list of output indices (default = all outputs)
        """
        if not isinstance(model, (Model, Sequential)):
            raise ValueError("Invalid model type. Must be a keras Model or Sequential.")

        self.model = model

        # determine output channels
        if outchannels is None:
            self.outchannels = list(range(model.output.shape[-1]))
            if verbose:
                print("Evaluated output channel(s): All")
        else:
            self.outchannels = outchannels
            if verbose:
                print("Evaluated output channels:", self.outchannels)

        if verbose:
            print("Integrated Gradients ready.")

    def explain(self, sample, outc=0, reference=None, num_steps=50, verbose=0):
        """
        sample: np.ndarray or list of np.ndarrays (input to the model)
        outc: output channel index
        reference: baseline (default = 0s with same shape as sample)
        num_steps: number of interpolation steps
        """
        if isinstance(sample, np.ndarray):
            sample = [sample]

        if reference is None:
            reference = [np.zeros_like(s) for s in sample]
        elif isinstance(reference, np.ndarray):
            reference = [reference]

        assert len(sample) == len(reference), "Sample and reference must match in length"

        # Build interpolation coefficients (exclude endpoint to mimic original IG paper approach)
        alphas = tf.linspace(0.0, 1.0, num_steps + 1)  # include endpoint
        # Remove the first baseline-only point when summing later (standard IG uses Riemann approximation)
        alphas = alphas[1:]

        interpolated_tensors = []
        for s, r in zip(sample, reference):
            s_t = tf.convert_to_tensor(s, dtype=tf.float32)
            r_t = tf.convert_to_tensor(r, dtype=tf.float32)
            # shape (num_steps, features)
            interp = r_t + tf.expand_dims(alphas, -1) * (s_t - r_t)
            interpolated_tensors.append(interp)

        # If single input, keep a tensor; else list of tensors
        model_inputs = interpolated_tensors[0] if len(interpolated_tensors) == 1 else interpolated_tensors

        with tf.GradientTape() as tape:
            # Watch concrete interpolated tensors (not symbolic placeholders)
            if isinstance(model_inputs, list):
                for inp in model_inputs:
                    tape.watch(inp)
            else:
                tape.watch(model_inputs)
            preds = self.model(model_inputs, training=False)  # shape (num_steps, output_dim)
            target = preds[:, outc]

        # Take gradients w.r.t. concrete tensors we watched
        if isinstance(model_inputs, list):
            gradients = tape.gradient(target, model_inputs)
        else:
            gradients = tape.gradient(target, [model_inputs])
        if not isinstance(gradients, (list, tuple)):
            gradients = [gradients]

        attributions = []
        # Average gradients (integral approximation) times (input - baseline)
        for i, (s, r, g) in enumerate(zip(sample, reference, gradients)):
            # g shape: (num_steps, features)
            avg_grad = tf.reduce_mean(g, axis=0)  # (features,)
            delta = tf.convert_to_tensor(s - r, dtype=tf.float32)
            attr = delta * avg_grad
            attributions.append(attr.numpy())

        if len(attributions) == 1:
            return attributions[0]
        return attributions
