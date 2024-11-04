import cv2
import argparse
from image_tools_api import ImageProcessor

if __name__ == "__main__":
    print("First task: stiching images into one\n\n\n")
    # Parsing
    parser = argparse.ArgumentParser(description="Create orthophoto from images in a directory")
    parser.add_argument("directory_path", type=str, help="Path to the directory with images")
    parser.add_argument("--scale_factor", default= 1, type=float, help="Factor by which to scale down images (e.g., 2 for half size)")
    args = parser.parse_args()
    # Declare api class
    image_processor = ImageProcessor.ImageProcessor()

    images = image_processor.load_images(args.directory_path, args.scale_factor)
    result = image_processor.stitch_images(images)
    if(result):
        cv2.imshow("Stitched Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(str(args.directory_path) + "/stitched.jpg", result)

        print("Saveing in: ", str(args.directory_path) + "/stitched.jpg")
    
    