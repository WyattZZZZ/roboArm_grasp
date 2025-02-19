from ultralytics import YOLO
import cv2 as cv

class Detection:
    def __init__(self, model_file_name):
        self.yolo = YOLO(model_file_name)

    def detect(self, image):
        return self.yolo(image)

    def display(self, image):
        results = self.detect(image)
        cv.drawContours(image, results, -1, (0, 255, 0), 2)

    def data(self, image):
        return self.detect(image)

    def calculate_center(self, result):
        pass

    def contours_label(self, result) -> list:
        pass

    def data_formatted(self, image):
        results = self.data(image)
        response = []
        for result in results:
            center = self.calculate_center(result)
            response.append(result.update({"center": center}))

        return response

    def make_mask(self, response, depth):
        data = self.contours_label(response)
        for i in range(len(depth)):
            for j in range(len(depth[0])):
                if data[i][j] == 0:
                    depth[i][j] = 100

        return depth