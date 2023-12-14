from ultralytics import YOLO
from os import path

base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

class Gender:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)

    def count_femoral_pores(self, results):
        mask_count = 0
        for r in results:
            if r.masks:
                mask_count += len(r.masks)
        return mask_count

    def analyze_image(self, image_path, save_dir, name):
        resultData = self.yolo_model.predict(source=image_path, save=True, imgsz=320, project=save_dir, name=name)

        # femoral pores 개수 세기
        count = self.count_femoral_pores(resultData)
        result = ''
        if count >5:
          result = f"천공이 {count}개가 확인이 됩니다. 수컷이 유력합니다."
        elif count > 3:
            result = f"천공이 {count}개가 확인이 됩니다. 수컷 추정입니다."
        else:
            result = f"천공이 {count}개가 확인이 됩니다. 암컷 추정 혹은 미구분입니다."

        return result

