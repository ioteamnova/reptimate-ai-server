import math

import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import extcolors
import os
from os import path
base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

class Lateral:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)

    def analyze_image(self, image_path, date, save_dir):
        image = cv2.imread(image_path)
        results = self.yolo_model.predict(source=image_path, save=False, imgsz=320)

        # YOLO 결과에서 윤곽선 좌표 추출
        contour_points = []

        for r in results:
            li = r.masks.xyn
            if len(li) > 0:
                # 좌표 비율을 이미지 크기에 맞게 조정
                img_height, img_width, _ = image.shape
                contour_points.extend(li[0] * np.array([img_width, img_height]))

        # contour_points를 정수형으로 변환
        contour_points = np.array(contour_points, dtype=np.int32)

        # 화면 비율 맞춰주는 부분
        for point in contour_points:
            point[0] = round(point[0])
            point[1] = round(point[1])

        # 원본 이미지 복사
        origin_image = image.copy()

        # 외곽 그리기
        ctr = contour_points.reshape((-1, 1, 2))
        contour_img = cv2.drawContours(image, [ctr], -1, (0, 0, 0), cv2.FILLED)

        # 원본 이미지 - 외곽 이미지 = 두 이미지의 차 구하기
        crop_img = cv2.absdiff(origin_image, contour_img)

        # 좌표 배열을 NumPy 배열로 변환
        contour_points = np.array(contour_points)

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
        cropImgPath = save_dir+date+'cropped_image.png'
        cv2.imwrite(cropImgPath, dst)
        img = Image.open(cropImgPath)

        # 색상과 픽셀 수 추출
        colors, pixel_count = extcolors.extract_from_image(img)

        # 색상을 밝은 순으로 정렬하고 조건을 만족하지 않는 항목 제거
        sorted_colors = sorted(colors, key=lambda x: np.sum(x[0]))
        sorted_colors = [color_info for color_info in sorted_colors if (color_info[1] / pixel_count) * 100 >= 1]
        rgb_values = [color_info[0] for color_info in sorted_colors]

        # 픽셀 백분율 계산
        total_percentage = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors])
        Second = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors[1:]])
        Third = sum([(color_info[1] / pixel_count) * 100 for color_info in sorted_colors[2:]])

        SecondPercent = math.floor(Second / total_percentage * 100)
        ThirdPercent = math.floor(Third / total_percentage * 100)

        score = 0

        if SecondPercent > 80:
            score = score + 65
        elif SecondPercent > 70:
            score = score + 60
        elif SecondPercent > 60:
            score = score + 55
        elif SecondPercent > 50:
            score = score + 50
        elif SecondPercent > 40:
            score = score + 45
        elif SecondPercent > 30:
            score = score + 40
        else:
            score = score + 30


        if ThirdPercent > 60:
            score = score + 50
        elif ThirdPercent > 50:
            score = score + 40
        elif ThirdPercent > 40:
            score = score + 30
        elif ThirdPercent > 30:
            score = score + 20
        elif ThirdPercent > 20:
            score = score + 10

        # 100점이 넘어가는 경우 100점 고정
        if score > 100:
            score = 100;

        os.remove(cropImgPath)
        os.remove(image_path)
        result = {
            "score": score,
            "SecondPercent": SecondPercent,
            "ThirdPercent": ThirdPercent,
            "RGB": rgb_values
        }
        return result
