#!/usr/bin/env python3
"""Module for the Yolo class using the Yolo v3 algorithm."""
import os
import numpy as np
import cv2
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
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))

            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h
            bw = (np.exp(t_w) * pw) / input_w
            bh = (np.exp(t_h) * ph) / input_h

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes based on class score threshold."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered bounding boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            mask = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[mask]
            cls_scores = box_scores[mask]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                intersection = (np.maximum(0, x2 - x1) *
                                np.maximum(0, y2 - y1))

                area_best = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                             (cls_boxes[0, 3] - cls_boxes[0, 1]))
                area_rest = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                             (cls_boxes[1:, 3] - cls_boxes[1:, 1]))

                union = area_best + area_rest - intersection
                iou = intersection / union

                keep = np.where(iou < self.nms_t)[0]
                cls_boxes = cls_boxes[keep + 1]
                cls_scores = cls_scores[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder."""
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for the Darknet model."""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with bounding boxes, class names, and scores."""
        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = round(float(box_scores[i]), 2)
            label = "{} {}".format(class_name, score)

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            os.makedirs('detections', exist_ok=True)
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Perform object detection on all images in a folder."""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i, image in enumerate(images):
            image_outputs = [output[i] for output in outputs]
            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs, image_shapes[i]
            )
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores
            )

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(image, boxes, box_classes, box_scores, file_name)

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
