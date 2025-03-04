import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def depth2pointcloud(depth_image, intrinsics, depth_scale=0.001):
    """Convert depth image to point cloud
    
    Args:
        depth_image: Depth image from RealSense camera
        intrinsics: Camera intrinsics
        depth_scale: Depth scale factor (default=0.001)
        
    Returns:
        o3d.geometry.PointCloud: Point cloud object
    """
    # Get image dimensions
    height, width = depth_image.shape
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    
    # Create arrays for storing 3D points
    points = []
    
    # Convert depth image to 3D points
    for v in range(height):
        for u in range(width):
            depth_value = depth_image[v, u] * depth_scale
            
            # Skip invalid depth values
            if depth_value == 0 or depth_value > 10:  # Maximum depth threshold (10m)
                continue
                
            # Deproject pixel to 3D point
            x = (u - intrinsics.ppx) / intrinsics.fx * depth_value
            y = (v - intrinsics.ppy) / intrinsics.fy * depth_value
            z = depth_value
            
            points.append([x, y, z])
    
    # Convert to numpy array and set as points in point cloud
    if points:
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    
    return point_cloud

class Detection:
    
    @staticmethod
    def read_mask(file_name):
        """Read mask image from file
        
        Args:
            file_name: Path to mask image file
            
        Returns:
            tuple: (mask_image, mask_type)
        """
        type_name = file_name.split('.')[0]  # Get mask type from filename
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
        return img, type_name
    
    
    @staticmethod
    def predict_bounding_box(point_cloud: o3d.geometry.PointCloud):
        """Predict oriented bounding box for point cloud
        
        Args:
            point_cloud: Open3D point cloud object
            
        Returns:
            o3d.geometry.OrientedBoundingBox: Oriented bounding box
        """
        # Ensure point cloud has enough points
        if len(point_cloud.points) < 4:
            return None
            
        # Compute oriented bounding box
        obb = point_cloud.get_oriented_bounding_box()
        
        return obb
    
    @staticmethod
    def calculate_volume(bbox):
        extent = bbox.extent()
        return extent[0] * extent[1] * extent[2]
    
    
    @staticmethod
    def find_volume(self, depth_image, mask_file=None, intrinsics=None, depth_scale=0.001):
        """Find volume of object in depth image
        
        Args:
            depth_image: Depth image from RealSense camera
            mask_file: Path to mask image file (default=None)
            intrinsics: Camera intrinsics (default=None)
            depth_scale: Depth scale factor (default=0.001)
            
        Returns:
            float: Volume of object in cubic meters
        """
        # Apply mask if provided
        if mask_file:
            mask, _ = Detection.read_mask(mask_file)
            if mask.shape != depth_image.shape:
                mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))
            
        # Create masked depth image by iterating through mask
            avg_depth = np.mean(depth_image)
            masked_depth = depth_image.copy()
            height, width = mask.shape
            for i in range(height):
                for j in range(width):
                    if mask[i, j] == 0:  # If pixel is masked
                        masked_depth[i, j] = avg_depth
            else:
                masked_depth = depth_image
            
        # Convert to point cloud
        point_cloud = depth2pointcloud(masked_depth, intrinsics, depth_scale)
        
        # Get 3D bounding box
        bbox_3d = Detection.predict_bounding_box(point_cloud)

        volume = Detection.calculate_volume(bbox_3d)
    
        return volume
        

class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30):
        """Initialize RealSense camera
        
        Args:
            width: Image width (default=640)
            height: Image height (default=480)
            fps: Frames per second (default=30)
        """
        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
    def get_rgbd_frame(self):
        """Get aligned RGB and depth frames
        
        Returns:
            tuple: (color_image, depth_image, intrinsics)
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        # 转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite("./yolo/inference/now.jpg", color_image)
        # 融合成 RGBD 图像：将深度作为第四个通道添加到彩色图上
        rgb_d_image = np.dstack((depth_image, depth_image, depth_image))
        return color_image, depth_image
        
    def get_streamed_depth_data(self):
        """Get streaming depth data
        
        Returns:
            tuple: (depth_image, intrinsics)
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Get depth frame
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            return None, None
            
        # Convert to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return depth_image, self.intrinsics
        
    def stop(self):
        """Stop the RealSense pipeline"""
        self.pipeline.stop()
