import cv2
import argparse
from image_tools_api import ImageProcessor


def save_results(results, output_file='results.txt'):
    with open(output_file, 'w') as f:
        for img_path, detections in results:
            f.write(f'Image: {img_path}\n')
            for label, conf, (x1, y1, x2, y2) in detections:
                f.write(f'  Object: {label}, Trust: {conf:.2f}, Bounds: {x1}, {y1}, {x2}, {y2}\n')
            f.write('\n')


if __name__ == "__main__":
    print("Third task: detections\n\n\n")

    results = []

    # Parsing
    parser = argparse.ArgumentParser(description="Third task")
    parser.add_argument("directory_path", type=str, help="(str) Path to the directory with images")
    parser.add_argument("--save_txt", default=False, type=bool, help="(bool) Save detections in txt file?")
    args = parser.parse_args()

    # Declare api class
    image_processor = ImageProcessor.ImageProcessor()
    image_files = image_processor.get_image_files(args.directory_path)
        
    for img_path in image_files:
        img = image_processor.load_image(img_path)

        detections, img_detect = image_processor.get_detections(img)

        

        # Сохранение результата для каждого изображения
        results.append((image_processor.get_file_name_without_extension(img_path), detections))
        
        # Шаг 3: Отображение результата
        cv2.imshow('Detections', img_detect)
        cv2.waitKey(0)  # Ждем нажатия клавиши для следующего изображения
    
    if(args.save_txt):
        print("Saveing in: ", str(args.directory_path) + "/results.txt")
        save_results(results, str(args.directory_path) + "/results.txt")

    cv2.destroyAllWindows()
    
