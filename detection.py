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
        
    # Todo: construct a dictionary with class names and their corresponding center.
    @staticmethod
    def construct_label_dict(self):
        pass
        
    @staticmethod
    def data_formatted(self, detections, rgb_image, depth_image, intrinsics):
        """Search
        
        Args:
            rgb_image: RGB image from RealSense camera
            depth_image: Depth image from RealSense camera
            intrinsics: Camera intrinsics
            
        Returns:
            dict: Dictionary with object information including 3D bounding boxes
        """
        # Convert depth image to point cloud
        point_cloud = depth2pointcloud(depth_image, intrinsics)
        
        # Group detections by class
        labeled_objects = Detection.construct_label_dict(detections)
        
        # Process each detection
        result = {}
        for class_name, objects in labeled_objects.items():
            result[class_name] = []
            
            for obj in objects:
                # Extract masked point cloud for this object
                x, y, w, h = obj['bbox']
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                x_end = min(rgb_image.shape[1], x + w)
                y_end = min(rgb_image.shape[0], y + h)
                
                # Create mask for this object
                mask = np.zeros_like(depth_image)
                mask[y:y_end, x:x_end] = 1
                
                # Apply mask to depth image
                masked_depth = depth_image.copy()
                masked_depth[mask == 0] = 0
                
                # Convert masked depth to point cloud
                obj_point_cloud = depth2pointcloud(masked_depth, intrinsics)
                
                # Skip if no points
                if not obj_point_cloud.points:
                    continue
                
                # Compute 3D bounding box
                bbox = self.predict_bounding_box(obj_point_cloud)
                if bbox:
                    result[class_name].append({
                        'center_2d': obj['center'],
                        'bbox_2d': obj['bbox'],
                        'bbox_3d': bbox,
                        'confidence': obj['confidence']
                    })
                    
        return result
    
    @staticmethod
    def read_mask(self, file_name):
        """Read mask image from file
        
        Args:
            file_name: Path to mask image file
            
        Returns:
            tuple: (mask_image, mask_type)
        """
        type_name = file_name.split('.')[0]  # Get mask type from filename
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None, type_name
            
        # Threshold to create binary mask
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        return img, type_name
    
    
    @staticmethod
    def predict_bounding_box(self, point_cloud: o3d.geometry.PointCloud):
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
    def calculate_volume(self, bbox: o3d.geometry.PointCloud):
        extent = bbox.extent();
        return extent[0] * extent[1] * extent[2]
        

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
        
        
    def data_formatted(self, BB, depth_image, mask_file=None):
        """Format data with masked depth image
        
        Args:
            BB: Bounding box coordinates [x, y, w, h]
            depth_image: Depth image
            mask_file: Path to mask file (optional)
            
        Returns:
            dict: Formatted data with masked depth
        """
        # Apply mask if provided
        if mask_file:
            mask, mask_type = self.read_mask(mask_file)
            if mask is not None:
                # Resize mask to match depth image if needed
                if mask.shape != depth_image.shape:
                    mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))
                
                # Calculate background average depth
                bg_mask = mask == 0
                bg_depth = depth_image[bg_mask]
                if len(bg_depth) > 0:
                    bg_avg = np.mean(bg_depth)
                else:
                    bg_avg = 0
                    
                # Apply mask: keep object, set background to average
                masked_depth = depth_image.copy()
                masked_depth[bg_mask] = bg_avg
            else:
                masked_depth = depth_image
        else:
            # Use bounding box as mask
            x, y, w, h = BB
            masked_depth = np.zeros_like(depth_image)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            x_end = min(depth_image.shape[1], x + w)
            y_end = min(depth_image.shape[0], y + h)
            
            # Create mask for this region
            masked_depth[y:y_end, x:x_end] = depth_image[y:y_end, x:x_end]
            
        # Convert to point cloud
        point_cloud = depth2pointcloud(masked_depth, self.intrinsics, self.depth_scale)
        
        # Get 3D bounding box
        detector = Detection("")  # Empty model path as we're just using the bounding box method
        bbox_3d = detector.predict_bounding_box(point_cloud)
        
        # Calculate center in 2D
        center_2d = self.get_BB_center(BB)
        
        result = {
            'bbox_2d': BB,
            'center_2d': center_2d,
            'point_cloud': point_cloud
        }
        
        if bbox_3d:
            result['bbox_3d'] = bbox_3d
            result['volume'] = bbox_3d.volume()
            result['dimensions'] = bbox_3d.extent
            
        return result
        
    def stop(self):
        """Stop the RealSense pipeline"""
        self.pipeline.stop()

# Main application
if __name__ == "__main__":
    