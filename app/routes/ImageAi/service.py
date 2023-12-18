from typing import List,Union
from fastapi import UploadFile, File, Depends
from sqlalchemy.orm import Session
from core.analyzer_lateral.lateral_predict import Lateral
from core.analyzer_top.topmode_predict import TopMode
from core.analyzer_gender.gender_predict import Gender
from core.analyzer_img_checker.img_checker import Img_checker
from utils.S3 import s3_uploader
from routes.ImageAi.schemas.ValueAnalyzer_schema import ValueAnalyzerSchema
from routes.ImageAi.schemas.MorphList_schema import MorphListSchema
from routes.ImageAi.dtos.ValueAnalyzer_dto import ValueAnalyzerCreate, ValueAnalyze, ValueAnalyzerResult
from os import path
import os
import datetime
from core.database.conn import db
from fastapi import HTTPException
import shutil
import json
from utils.color_utils import find_similar_colors, rgb2lab
from utils.linebreeding_utils import morph_re_selection, score_compare_selection, make_morph_explanation, sort_feature_order


base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
class image_ai_service:

    # 가치판단 진행
    async def assess_value(self, data: ValueAnalyze, files: List[UploadFile]):
        topAnalyzer = TopMode(base_dir + '/core/analyzer_top/datasets/train/weights/best.pt')
        lateralAnalyzer = Lateral(base_dir + '/core/analyzer_lateral/datasets/train/weights/best.pt')
        img_checker = Img_checker(base_dir + '/core/analyzer_img_checker/datasets/train/weights/best.pt')
        save_dir = base_dir + '/core/analyzer_lateral/datasets/test/images'
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = ''
        topImgPath = ''
        leftLateralImgPath = ''
        righLateraltImgPath = ''

        # 이미지를 순회하면서 각각 원본 이름으로 저장
        for idx, file in enumerate(files):
            # 현재 시간을 포함한 파일 이름 생성
            file_name_without_extension = f"{current_time}_{file.filename.rsplit('.', 1)[0]}"
            if idx == 0:  # 탑 이미지
                file_name = os.path.join(save_dir, f"{file_name_without_extension}_topImgPath.jpeg")
                topImgPath = file_name
            elif idx == 1:  # 왼쪽 레터럴 이미지
                file_name = os.path.join(save_dir, f"{file_name_without_extension}_leftLateralImgPath.jpeg")
                leftLateralImgPath = file_name
            elif idx == 2:  # 오른쪽 레터럴 이미지
                file_name = os.path.join(save_dir, f"{file_name_without_extension}_righLateraltImgPath.jpeg")
                righLateraltImgPath = file_name
            # 파일을 지정된 디렉토리에 저장
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, "wb") as f:
                await file.seek(0)
                contents = await file.read()
                f.write(contents)


        # 이미지가 크레스티드 게코인지 확인 - 현재는 gecko 클래스 탐지하면 통과하함
        img_checker.img_checking(topImgPath)
        img_checker.img_checking(leftLateralImgPath)
        img_checker.img_checking(righLateraltImgPath)

        # 머리, 등, 꼬리 검사
        try:
            print("!!!!!!!")
            topResult = topAnalyzer.analyze_image(topImgPath, current_time + "_top_", save_dir)
            print("!!!!!!!")

            print(topResult)
        except Exception as e:
            # 예외 처리
            error_message = 'Top Part Error'
            return {"error": error_message}
        # 왼쪽 레터럴 검사
        try:
            leftResult = lateralAnalyzer.analyze_image(leftLateralImgPath, current_time + "_left_", save_dir)
            print(leftResult)
        except Exception as e:
            # 예외 처리
            error_message = str(e)
            return {"error": error_message}
        # 오른쪽 레터럴 검사
        try:
            rightResult = lateralAnalyzer.analyze_image(righLateraltImgPath, current_time + "_right_", save_dir)
            print(rightResult)
        except Exception as e:
            # 예외 처리
            error_message = str(e)
            return {"error": error_message}

        total_score = (leftResult.get('score') + rightResult.get('score') + topResult.get("haed_score") + topResult.get(
            "tail_score") + topResult.get("dorsal_score")) / 5

        # 결과 저장
        result = ValueAnalyzerCreate.updateFrom(None, 'auto_save', data.morph, data.gender, topResult.get("haed_score"),
                                            topResult.get("dorsal_score"), topResult.get("tail_score"),
                                            leftResult.get('score'),rightResult.get('score'),
                                            total_score, leftResult, rightResult)
        return result

    # 가치판단 진행할때 데이터를 저장해주는 기능
    async def analyzer_auto_save(
            self,
            result: ValueAnalyzerCreate,
            files: List[UploadFile] = File(...),
            session: Session = Depends(db.session)):
        # s3_uploader를 사용하여 이미지 업로드
        for idx, file in enumerate(files):
            uploaded_image = s3_uploader.upload_image(file, 'ImageAi')
            image_url = uploaded_image['message'].split('URL: ')[1]
            # idx에 따라 필드 설정
            if idx == 0:
                result.top_img = image_url
            elif idx == 1:
                result.left_img = image_url
            elif idx == 2:
                result.right_img = image_url

        value_analyzer = ValueAnalyzerSchema(**result.dict())  # ValueAnalyzete 모델의 데이터를 ImageAi 모델로 변환rCrea
        session.add(value_analyzer)
        session.commit()
        session.refresh(value_analyzer)
        result2 = ValueAnalyzerResult.updateFrom(6, value_analyzer)
        result2.idx = value_analyzer.idx
        result2.top_img = result.top_img
        result2.left_img = result.left_img
        result2.right_img = result.right_img
        return result2

    # 가치판단에서 저장하지 않고 데이터 리턴을 위해 정제해주는 기능
    async def analyzer_refine_data(
            self,
            result: ValueAnalyzerCreate,
            files: List[UploadFile] = File(...)):
        # s3_uploader를 사용하여 이미지 업로드
        for idx, file in enumerate(files):
            uploaded_image = s3_uploader.upload_image(file, 'ImageAi')
            image_url = uploaded_image['message'].split('URL: ')[1]
            # idx에 따라 필드 설정
            if idx == 0:
                result.top_img = image_url
            elif idx == 1:
                result.left_img = image_url
            elif idx == 2:
                result.right_img = image_url

        return result

    # 가치판단 데이터가 저장된 상태에서 idx와 petname을 수정해주는 기능
    async def analyzer_save(
            self,
            idx: int,
            userIdx: int,
            petName: str,
            session: Session = Depends(db.session)):

        value_analyzer = session.query(ValueAnalyzerSchema).filter(ValueAnalyzerSchema.idx == idx).first()

        # 해당하는 idx의 행이 없으면 예외 처리
        if not value_analyzer:
            raise HTTPException(status_code=404, detail="Value Analyzer not found")

        # userIdx와 petName 업데이트
        value_analyzer.user_idx = userIdx
        value_analyzer.pet_name = petName

        # 변경사항 커밋
        session.commit()

    # 암수 구분하는 기능
    async def gender_discrimination(
            self,
            file: UploadFile):
        # 훈련 데이터 파일 경로를 넣어주며 클래스 생성
        genderAnalyzer = Gender(base_dir + '/core/analyzer_gender/datasets/train/weights/best.pt')
        # 받은 이미지 저장할 경로 설정
        save_dir = base_dir + '/core/analyzer_gender/datasets/test/images'
        # 저장할 이미지 이름에 현제 날짜와 시간을 넣어 주기 위한 날짜 & 시간
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 이름 합치기 & 가공
        file_full_name = f"{current_time}_{file.filename.rsplit('.', 1)[0]}"
        # 최종 파일 이름
        file_name = os.path.join(save_dir, f"{file_full_name}_genderImgPath.jpeg")
        # 세그멘테이션 결과 이미지 경로
        result_path = f"{save_dir}/{file_full_name}/{file_full_name}_genderImgPath.jpeg"
        # 이미지 저장
        with open(file_name, "wb") as f:
            await file.seek(0)
            # 파일을 먼저 읽은 다음에 저장
            contents = await file.read()
            f.write(contents)
        # 본격적으로 성별 구분 기능 실행. 저장된 이미지 경로와 결과물 저장할 이름 만들 떄 필요한 값들 넣어줌
        genderResult = genderAnalyzer.analyze_image(file_name, save_dir, file_full_name)

        # # 세그멘테이션 결과 이미지 file로 객체화
        with open(result_path, "rb") as sagFile:
            resultFile = sagFile.read()
        # 다쓴 이미지는 삭제, 영구 저장할 이미지는 S3에 저장함
        returnImg = s3_uploader.upload_local_image(resultFile, 'gender_discrimination', f"{file_full_name}_gender_result.jpeg")
        # 세그멘테이션 결과물 디렉토리 삭제
        dir_path = f"{save_dir}/{file_full_name}"
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error: {e.strerror}")

        # 원본 이미지 삭제
        os.remove(file_name)
        result = {"result": genderResult, "returnImg": returnImg['message'].split('URL: ')[1]}
        return result

    #도마뱀 라인 브리딩 분석 진행하는 기능
    async def get_analyzer_data(  # 분석 진행하기
            self,
            UserValueAnalyzer: ValueAnalyzerCreate,
            session: Session = Depends(db.session)):

        # 왼 오 중 더 잘나온 걸로 선택할 것.
        left_info_object = json.loads(UserValueAnalyzer.left_info)
        right_info_object = json.loads(UserValueAnalyzer.right_info)

        left_feature_order = self.make_feature_order_list(left_info_object)
        right_feature_order = self.make_feature_order_list(right_info_object)

        print("left_feature_order")
        print(left_feature_order)
        print("right_feature_order")
        print(right_feature_order)

        final_direction_leteral = ""
        feature_order = ""

        # 왼쪽 오른쪽 레터럴 크기 비교 - 더 상태가 좋은걸로 평가하기 위한 비교
        if left_feature_order[0][0] == right_feature_order[0][0]:
            if left_feature_order[1][0] == right_feature_order[1][0]:
                if left_feature_order[0][0] == 1 and right_feature_order[0][0] == 1: #1차 형질은 클수록 좋지 않다고 판단
                    if left_feature_order[0][2] > right_feature_order[0][2]:
                        final_direction_leteral = "right"
                    elif left_feature_order[0][2] < right_feature_order[0][2]:
                        final_direction_leteral = "left"
                    else:
                        final_direction_leteral = "left" # 같으면 왼쪽
                elif left_feature_order[0][0] == 2 and right_feature_order[0][0] == 2: #2차 형질은 클수록 좋다고 판단
                    if left_feature_order[0][2] > right_feature_order[0][2]:
                        final_direction_leteral = "left"
                    elif left_feature_order[0][2] < right_feature_order[0][2]:
                        final_direction_leteral = "right"
                    else:
                        final_direction_leteral = "left"  # 같으면 왼쪽
                elif left_feature_order[0][0] == 3 and right_feature_order[0][0] == 3: #3차 형질은 클수록 좋다고 판단
                    if left_feature_order[0][2] > right_feature_order[0][2]:
                        final_direction_leteral = "left"
                    elif left_feature_order[0][2] < right_feature_order[0][2]:
                        final_direction_leteral = "right"
                    else:
                        final_direction_leteral = "left"  # 같으면 왼쪽
            elif left_feature_order[1][0] > right_feature_order[1][0]:
                final_direction_leteral = "left"
            elif left_feature_order[1][0] < right_feature_order[1][0]:
                final_direction_leteral = "right"
        elif left_feature_order[0][0] > right_feature_order[0][0]:
            final_direction_leteral = "left"
        elif left_feature_order[0][0] < right_feature_order[0][0]:
            final_direction_leteral = "right"

        #최종 평가 방향 선택
        if final_direction_leteral == "left":
            feature_order = left_feature_order
        elif final_direction_leteral == "right":
            feature_order = right_feature_order

        # 분석 결과 데이터 전부 가져오기
        ValueAnalyzer_datas = await self.get_analyzer_all_data(session)  # assess_value 메서드 호출

        # feature_order의 순서에서 첫번째 데이터 수집
        search_list = morph_re_selection(ValueAnalyzer_datas, feature_order, final_direction_leteral, [], 1)

        #유사도 임계값
        threshold = 30  # Adjust the threshold to control similarity tolerance

        # 가상 유사도가 높은 컬러의 개체 추출
        similar_datas = find_similar_colors(feature_order[0][1], search_list, threshold)
        print("Similar colors _1:", similar_datas)

        if feature_order[1][1] != 0:  # 유저 RGB 순서 데이터에서 빈 값이 있으면 추가하지 않는다.
            # 두번째로 큰 형질 리스트 수집
            search_list = morph_re_selection(ValueAnalyzer_datas, feature_order, final_direction_leteral, similar_datas[1], 2)

            # 가상 유사도가 높은 컬러의 개체 추출
            similar_datas = find_similar_colors(feature_order[1][1], search_list, threshold)
            print("Similar colors _2:", similar_datas)

        if feature_order[2][1] != 0:  # 유저 RGB 순서 데이터에서 빈 값이 있으면 추가하지 않는다.
            # 세번째로 큰 형질 리스트 수집
            search_list = morph_re_selection(ValueAnalyzer_datas, feature_order, final_direction_leteral, similar_datas[1], 3)

            # 가상 유사도가 높은 컬러의 개체 추출
            similar_datas = find_similar_colors(feature_order[2][1], search_list, threshold)
            print("Similar colors _3:", similar_datas)

        # 네번째로 레터럴, 도살 점수로 리스트 수집
        # 이 중에서 레터럴점수, 도살 점수, 다른 성별 의 조건으로 서치
        # 레터럴 점수가 같거나, 도살 점수가 같거나 클 경우 만 리스트업
        search_list = score_compare_selection(ValueAnalyzer_datas, UserValueAnalyzer, similar_datas[1])
        print("final_select - 레터럴점수, 도살 점수, 다른 성별 의 조건으로 서치")
        print(search_list)

        if len(search_list) != 0: # 추천 개체가 있는 경우
            # 모프 가치 결과로 모프 설명 만들기
            explan_data = make_morph_explanation(UserValueAnalyzer, feature_order)

            # morph 추천 리스트
            morph_recommend_list = await self.get_one_morph_condition(UserValueAnalyzer.morph, session)
            if len(morph_recommend_list) != 0:
                # 데이터 합치기
                result = {
                    "recommend_data":{
                        "morph": ValueAnalyzer_datas[search_list[0]].morph,
                        "gender": ValueAnalyzer_datas[search_list[0]].gender,
                        "top_img": ValueAnalyzer_datas[search_list[0]].top_img,
                        "left_img": ValueAnalyzer_datas[search_list[0]].left_img,
                        "right_img": ValueAnalyzer_datas[search_list[0]].right_img,
                    },
                    "explanation": explan_data,
                    "morph_recommend_list": morph_recommend_list[0].morph_recommend,
                }
            else:
                result = "morph 이름이 잘못 되었습니다. "
        else: # 추천 개체가 없을 경우
            result = "추천 가능한 개체가 없습니다. "

        return result

    async def get_analyzer_all_data( #분석 데이터 전부 불러오기
            self,
            session: Session = Depends(db.session)):

        ValueAnalyzer_datas = session.query(ValueAnalyzerSchema).all()  # User 객체와 User 중 이름만을 select함

        return ValueAnalyzer_datas

    async def get_one_morph_condition( #모프 종류, 추천 모프 데이터 가져오기
            self,
            morph_name,
            session: Session = Depends(db.session)):

        morphListSchema_datas = session.query(MorphListSchema).filter(MorphListSchema.name == morph_name).all()

        return morphListSchema_datas

    #형질 순서 리스트를 민듬
    def make_feature_order_list(self, info_object):
        # 1. 레터럴 점수(왼, 오), 도살 점수, 비슷한 색상(어떻게 구할 것인가?) 다른 성별을 조건으로 검색함.
        first_feature = 100 - info_object['SecondPercent']
        second_feature = info_object['SecondPercent']
        third_feature = info_object['ThirdPercent']

        feature_order = []
        # #모프 색에 맞게 순서 행렬에 넣어주기
        if len(info_object['RGB']) >= 3:
            feature_order = [[1, rgb2lab(info_object['RGB'][0]), first_feature],
                             [2, rgb2lab(info_object['RGB'][1]), second_feature],
                             [3, rgb2lab(info_object['RGB'][2]), third_feature]]
        elif len(info_object['RGB']) == 2:
            feature_order = [[1, rgb2lab(info_object['RGB'][0]), first_feature],
                             [2, rgb2lab(info_object['RGB'][1]), second_feature],
                             [3, 0, 0]]
        elif len(info_object['RGB']) == 1:
            feature_order = [[1, rgb2lab(info_object['RGB'][0]), first_feature],
                             [2, 0, 0],
                             [3, 0, 0]]

        feature_order = sort_feature_order(feature_order)
        return feature_order