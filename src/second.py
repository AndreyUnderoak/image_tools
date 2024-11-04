import cv2
import argparse
from image_tools_api import ImageProcessor


if __name__ == "__main__":
    print("Second task: preprocess images\n\n\n")

    # Parsing
    parser = argparse.ArgumentParser(description="Second task")
    parser.add_argument("directory_path", type=str, help="(str) Path to the directory with images")
    parser.add_argument("--scale_factor", default=1, type=float, help="(float) Factor by which to scale down images")
    parser.add_argument("--contrast_method", default="clahe", type=str, help="(str) Contrast enhancement method (\"hist_eq\" for histogram equalization, \"clahe\" for CLAHE)")
    parser.add_argument("--white_balance", default=True, type=bool, help="(bool) Flag to enable white balance correction")
    parser.add_argument("--clip_limit", default=3.0, type=float, help="(float) Clip limit for CLAHE.")
    parser.add_argument("--tile_grid_size_1", default=8,  type=int, help="(int) Tile grid size for CLAHE (tile_grid_size_1, tile_grid_size_2)")
    parser.add_argument("--tile_grid_size_2", default=8, type=int, help="(int) Tile grid size for CLAHE (tile_grid_size_1, tile_grid_size_2)")
    args = parser.parse_args()
    # Declare api class
    image_processor = ImageProcessor.ImageProcessor()
    image_files = image_processor.get_image_files(args.directory_path)

    print("Saveing in")
    for i in image_files:
        image = image_processor.load_image(i, args.scale_factor)
        processed_image = image_processor.preprocess_image(
            image, args.contrast_method, args.white_balance, args.clip_limit, tile_grid_size=(args.tile_grid_size_1, args.tile_grid_size_2)
        )

        cv2.destroyAllWindows()
        
        name = str(str(args.directory_path) + "_processed/"+image_processor.get_file_name_without_extension(i)+"_processed.jpg")
        print(name)
        image_processor.save_image(name, processed_image)
    print("Done")
    