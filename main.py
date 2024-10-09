from utils import (
    detect_license_plate,
    get_folder_images,
    get_text_from_image,
    fix_perspective,
    process_folder,
    process_results,
    extract_occurences,
    pick_top
)

def main(model_path, folder_path):
    results = process_folder(folder_path, model_path)
    processed = process_results(results)

    if len(processed) > 1:
        possible_license_plate = extract_occurences(processed)
    else:
        possible_license_plate = processed

    final_result = pick_top(possible_license_plate)

    return final_result

if __name__ == '__main__':
    model_path = './best.pt'
    folder_path = './data/2024000051'

    res = main(model_path, folder_path)
    print(res)