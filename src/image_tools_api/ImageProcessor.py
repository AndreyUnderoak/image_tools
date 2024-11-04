import cv2  
import numpy as np 
import os  
from ultralytics import YOLO  

class ImageProcessor:
    
    model = YOLO("yolov8n.pt")  
    stitcher = cv2.Stitcher_create()  
        
    @staticmethod
    def preprocess_image(image, contrast_method="hist_eq", white_balance=True, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Processes the image to enhance contrast and color balance.
        
        Parameters:
        - image (cv.image): Input image to be processed.
        - contrast_method (str): Contrast enhancement method ("hist_eq" for histogram equalization, "clahe" for CLAHE).
        - white_balance (bool): Flag to enable white balance correction.
        - clip_limit (float): Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).
        - tile_grid_size (tuple): Tile grid size for CLAHE.
        
        Returns:
        - image : Processed image.
        """
        
        if contrast_method == "hist_eq":
            # Check if the image is colored 
            if len(image.shape) == 3 and image.shape[2] == 3: 
                # Convert image from BGR to YUV 
                yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                # Equalize the histogram of the Y 
                yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
                # Convert back to BGR color space
                image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            else:
                # For grayscale images, apply histogram equalization directly
                image = cv2.equalizeHist(image)
        
        elif contrast_method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            # Check if the image is colored
            if len(image.shape) == 3 and image.shape[2] == 3:  
                
                lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
               
                lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
                
                image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            else:
                # CLAHE to grayscale images
                image = clahe.apply(image)
        
        # Apply white balance correction if enabled
        if white_balance:
            # Convert image to LAB 
            result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Calculate average values for the a and b channels
            avg_a = np.mean(result[:, :, 1])
            avg_b = np.mean(result[:, :, 2])
            # Adjust LAB channels based on avg
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            image = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return image  # Return the processed image
    
    @staticmethod
    def stitch_images(images):
        """
        Stitches together a list of images into a single panorama.
        
        Parameters:
        - images (list): List of images to stitch.
        
        Returns:
        - stitched_image (np.array): The stitched image if successful, None otherwise.
        """
        # Attempt to stitch the images together
        status, stitched_image = ImageProcessor.stitcher.stitch(images)
        # Check if stitching was successful
        if status == cv2.Stitcher_OK:  
            print("Stitch Success") 
            return stitched_image 
        print("Sorry, can't stitch these images. Try another...") 
        return None  
    
    @staticmethod
    def get_detections(img):
        """
        Gets object detections from an image using the YOLO model.
        
        Parameters:
        - img (np.array): Input image for detection.
        
        Returns:
        - result_data (list): List of detection results (label, confidence, bounding box).
        - img (np.array): Image with drawn bounding boxes.
        """
        model = ImageProcessor.model  
        detections = model(img)  
            
        # Process detection results
        result_data = []  
        for result in detections:  
            boxes = result.boxes  
            for box in boxes: 
                # Extract coordinates, confidence, and class index
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
                conf = box.conf[0].cpu().numpy()  
                cls = int(box.cls[0].cpu().numpy())  
                label = model.names[cls]  
                # Append the detection information to the results list
                result_data.append((label, conf, (x1, y1, x2, y2)))

                # Draw bounding box on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  
                cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
        return result_data, img  
    
    @staticmethod
    def load_images(directory_path, scale_factor=1):
        """
        Loads images from a specified directory and resizes them.
        
        Parameters:
        - directory_path (str): Path to the directory containing images.
        - scale_factor (float): Factor by which to resize images (1 = original size).
        
        Returns:
        - images (list): List of loaded and resized images.
        """
        images = []
        for filename in os.listdir(directory_path): 
            # Check if the file is an image based on its extension
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(directory_path, filename)  
                image = cv2.imread(file_path)  
                if image is not None: 
                    print("Loading image ", filename) 
                    # Resize image based on scale factor
                    new_width = int(image.shape[1] / scale_factor)
                    new_height = int(image.shape[0] / scale_factor)
                    resized_image = cv2.resize(image, (new_width, new_height))  
                    images.append(resized_image)  
                else:
                    print(f"Warning: Unable to load image {file_path}") 
        return images  
    
    @staticmethod
    def load_image(path, scale_factor=1):
        """
        Loads a single image from a specified path and resizes it.
        
        Parameters:
        - path (str): Path to the image file.
        - scale_factor (float): Factor by which to resize the image (1 = original size).
        
        Returns:
        - resized_image (np.array): Resized image if loaded successfully, None otherwise.
        """
        print("Loading image ", path) 
        image = cv2.imread(path)  
        if image is not None: 
            # Resize the image based on scale factor
            new_width = int(image.shape[1] / scale_factor)
            new_height = int(image.shape[0] / scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height)) 
        else:
            print(f"Warning: Unable to load image {path}") 
        return resized_image  
    
    @staticmethod
    def get_image_files(directory):
        """
        Retrieves a list of image files from a specified directory.
        
        Parameters:
        - directory (str): Path to the directory to search for images.
        
        Returns:
        - image_files (list): List of image file paths.
        """
        # Set of allowed image file extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
        image_files = []  

        for filename in os.listdir(directory):  
            file_path = os.path.join(directory, filename) 

            # Check if the file is an image based on its extension
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
                print("Identify image ", file_path)  
                image_files.append(file_path)  
    
        return image_files  
    
    @staticmethod
    def get_file_name_without_extension(file_path):
        """
        Extracts the file name without its extension from a file path.
        
        Parameters:
        - file_path (str): Path to the file.
        
        Returns:
        - file_name (str): Name of the file without its extension.
        """
        # Get the base file name with extension
        file_name_with_extension = os.path.basename(file_path)
        # Split the file name from its extension
        file_name, _ = os.path.splitext(file_name_with_extension)
        return file_name  

    @staticmethod
    def save_image(file_name, image):
        """
        Saves an image to a specified file path.
        
        Parameters:
        - file_name (str): Path where the image will be saved.
        - image (np.array): Image to be saved.
        """
        # Create the directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        # Save the image to the specified path
        cv2.imwrite(file_name, image)
