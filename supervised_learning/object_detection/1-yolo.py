#!/usr/bin/env python3
"""Module for the Yolo class using the Yolo v3 algorithm."""
import numpy as np
import tensorflow as tf


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detection model."""

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process the outputs from the Darknet model."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        # input dimensions of the model (e.g. 416x416)
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract raw values
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Build cx, cy grids (cell offsets)
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))

            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            # Anchor dimensions for this output scale
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Decode box center and dimensions (normalized to input size)
            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h
            bw = (np.exp(t_w) * pw) / input_w
            bh = (np.exp(t_h) * ph) / input_h

            # Convert to (x1, y1, x2, y2) relative to original image size
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Box confidence: sigmoid of index 4
            box_confidences.append(
                1 / (1 + np.exp(-output[..., 4:5]))
            )

            # Class probabilities: sigmoid of indices 5 onwards
            box_class_probs.append(
                1 / (1 + np.exp(-output[..., 5:]))
            )

        return boxes, box_confidences, box_class_probs
