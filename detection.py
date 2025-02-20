from ultralytics import YOLO
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import cv2


def depth2pointcloud(depth):
    pass

def calibrate(self, image_data, BB_data):
    pass

def if_masked(self, response, pixel):
    pass

def make_mask(self, response, depth):
    new = []
    for row in depth:
        c = sum(row)
        new.append(c)
    average = sum(new) / (len(depth)*len(depth[0]))
    for i in range(len(depth)):
        for j in range(len(depth[0])):
            if if_masked(response, depth[i][j]):
                depth[i][j] = average

    return depth


class Detection:
    def __init__(self, model_file_name):
        self.yolo = YOLO(model_file_name)

    def detect_test(self, image):
        d = self.yolo(image)
        image = cv.drawContours(image, d, -1, (0, 255, 0), 2)
        cv2.imshow("image", image)

    def detect(self, image):
        return self.yolo(image)

    def data(self, image):
        return self.detect(image)

    def calculate_center(self, result):
        pass

    def contours_label(self, result):
        pass

    def data_formatted(self, image):
        results = self.data(image)
        response = []
        for result in results:
            response.append(result.update({"center": self.calculate_center(result)}))

        return response


class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30):
        # 创建管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)


    def get_rgbd_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        # 转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 融合成 RGBD 图像：将深度作为第四个通道添加到彩色图上
        rgbd_image = np.dstack((color_image, depth_image))
        return color_image, depth_image, rgbd_image

    def get_BB_center(self, BB_list):
        pass

    def predict_bounding_box(self, color_image):
        pass

    def data_formatted(self, BB, center):
        pass

    def stop(self):
        self.pipeline.stop()


# 示例：启动采集并显示融合的 RGBD 图像和检测到的 bounding box
if __name__ == "__main__":
    rs_device = RealSenseRGBD()
    try:
        while True:
            pass
    finally:
        rs_device.stop()
        cv2.destroyAllWindows()