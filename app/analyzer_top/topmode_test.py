import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import extcolors
import os
from os import path
base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

# test 진행 후 이미지 뽑아내는 부분
class TopMode:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)

    def analyze_image(self, image_path, date):
        image = cv2.imread(image_path)
        results = self.yolo_model.predict(source=image_path, save=False, imgsz=320)

        # 부위별 유무 확인
        head_exist = 0
        dorsal_exist = 0
        tail_exist = 0

        for result in results:
            uniq, cnt = np.unique(result.boxes.cls.cpu().numpy(), return_counts=True)
            uniq_cnt_dict = dict(zip(uniq, cnt))

            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            li = masks.xyn


            # 클래스의 좌표마다 이미지로 만들기
            num = 0
            for c in result.boxes.cls:
                if int(c) == 0:
                    head_exist = 1
                elif int(c) == 1:
                    dorsal_exist = 1
                else:
                    tail_exist = 1

                # 화면 비율 맞춰주는 부분
                for point in li[num]:  # , (x, y)
                    point[0] = round(point[0] * boxes.orig_shape[1])
                    point[1] = round(point[1] * boxes.orig_shape[0])

                # 원본 이미지 복사
                origin_image = image.copy()

                # 외곽 그려주는 함수
                ctr = np.array(li[num]).reshape((-1, 1, 2)).astype(np.int32)
                # img = cv2.drawContours(image, [ctr], 0, (255), cv2.FILLED)
                contour_img = cv2.drawContours(image, [ctr], -1, (0, 0, 0), cv2.FILLED)

                # 원본 이미지 - 외곽 이미지 = 두 이미지의 차 구하기
                crop_img = cv2.absdiff(origin_image, contour_img)

                # 좌표 배열을 NumPy 배열로 변환
                contour_points = np.array(li[num])

                # 좌표 배열을 감싸는 최소한의 사각형을 구함
                x, y, w, h = cv2.boundingRect(contour_points)

                # 이미지에서 윤곽선 좌표에 해당하는 영역을 잘라냄
                cropped_image = crop_img[y:y + h, x:x + w]

                # 검정 부분 투명으로 처리하는 코드
                tmp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
                b, g, r = cv2.split(cropped_image)
                rgba = [b, g, r, alpha]
                dst = cv2.merge(rgba, 4)
                cropImgPath = base_dir + '/app/analyzer_lateral/datasets/test/images/' + date + 'cropped_image_' + self.yolo_model.names[int(c)] + ".png"
                cv2.imwrite(cropImgPath, dst)
                num = num + 1

        # 머리와 도살을 잡았을때만 진행 시킴
        if head_exist == 1 and dorsal_exist == 1:
            top_result = {} # 전체 결과 값
            #도살 - 색 분석, 예외처리
            # 이미지 열기
            dorsal_img = Image.open(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "cropped_image_dorsal.png")
            #색 분석하여 형질 추출 함수
            result = self.color_output(dorsal_img)
            #2차형질로 계산
            second_percent = result['second']
            dorsal_score = 0
            if second_percent >= 98:
                dorsal_score = 100
            elif second_percent >= 95 and second_percent < 98:
                dorsal_score = 90
            elif second_percent >= 92 and second_percent < 95:
                dorsal_score = 80
            elif second_percent >= 88 and second_percent < 92:
                dorsal_score = 75
            elif second_percent >= 80 and second_percent < 88:
                dorsal_score = 70
            elif second_percent >= 70 and second_percent < 80:
                dorsal_score = 60
            elif second_percent < 70:
                dorsal_score = 50

            top_result["dorsal_second_percent"] = second_percent
            top_result["dorsal_score"] = dorsal_score


            #머리 - (세로/가로)X100, 예외처리
            # 이미지 열기
            head_img = Image.open(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "cropped_image_head.png")

            width, height = head_img.size
            # 이미지 크기 (세로/가로) X 100
            head_percent = (height/width)*100
            haed_score = 0
            if head_percent >= 85:
                haed_score = 100
            elif head_percent >= 80 and head_percent < 85:
                haed_score = 90
            elif head_percent >= 70 and head_percent < 80:
                haed_score = 80
            elif head_percent >= 65 and head_percent < 70:
                haed_score = 70
            elif head_percent < 65:
                haed_score = 60
            top_result["head_percent"] = head_percent
            top_result["haed_score"] = haed_score


            #꼬리 - 유무
            tail_score = 0
            if tail_exist == 1:
                tail_score = 100
            else:
                tail_score = 50

            top_result["tail_score"] = tail_score
            os.remove(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "topImgPath.jpeg")
            os.remove(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "cropped_image_dorsal.png")
            os.remove(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "cropped_image_head.png")
            os.remove(base_dir + "/app/analyzer_lateral/datasets/test/images/" + date + "cropped_image_tail.png")

            return top_result
       #다시 찍으셈
        else:
            return 0

    # 색 추출 후 퍼센트 가져오는 함수
    def color_output(self, dorsal_img):
        # 색상과 픽셀 수 추출
        # colors, pixel_count = extcolors.extract_from_image(img)

        # 색상과 픽셀 수 추출
        colors, pixel_count = extcolors.extract_from_image(dorsal_img)

        # 색상을 밝은 순으로 정렬하고 조건을 만족하지 않는 항목 제거
        sorted_colors = sorted(colors, key=lambda x: np.sum(x[0]))

        sorted_colors = [color_info for color_info in sorted_colors if (color_info[1] / pixel_count) * 100 >= 1]

        # 픽셀 백분율 계산
        total_percentage = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors])
        Second = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors[1:]])
        Third = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors[2:]])

        SecondPercent = Second / total_percentage * 100

        result = {
            "first": total_percentage,
            "second": SecondPercent if SecondPercent >= 1 else "null",
        }
        return result

