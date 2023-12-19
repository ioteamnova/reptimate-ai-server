from typing import List,Union
from fastapi import UploadFile, File, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from utils.S3 import s3_uploader
from routes.TextAi.schemas.ChattingBot_schema import ChattingBotSchema
# from routes.TextAi.dtos.ChattingBot_dto import ValueAnalyzerCreate, ValueAnalyze
from os import path
import os
import datetime
from core.database.conn import db
from fastapi import HTTPException
import shutil
import json
import re
from konlpy.tag import Okt
Okt = Okt()

base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
import fasttext

class text_ai_service:

    async def transfrom_request_question(self, request_data):

        trans_data = request_data

        #한글만 남기고 다 지울것
        trans_data = re.sub(r"[^\uAC00-\uD7A3\s]", "", trans_data)

        # 크레스티가 있는 경우
        if trans_data.find('크레스티') >= -1:
            if trans_data.find('크레스티드 게코') > -1:
                trans_data = trans_data.replace('크레스티드 게코', '크레')
                print("크레스티드 게코 찾음")
            elif trans_data.find('크레스티드') > -1:
                trans_data = trans_data.replace('크레스티드', '크레')
                print("크레스티드 찾음")
            elif trans_data.find('크레스티') > -1:
                trans_data = trans_data.replace('크레스티', '크레')
                print("크레스티 찾음")

        #문장을 어절로 나누어서 여러개로 나눈다.
        trans_arr = Okt.phrases(trans_data)

        return trans_arr

    #챗봇 답변 주는 기능
    async def response_chatting_bot(self, request_data_arr):
        model = fasttext.load_model(base_dir + "/core/chatting_bot_model/model_cooking_reptile.bin")

        #어절로 나누어서 가져온 리스트를 하나씩 모델에 예측을 하여 가까운 퍼센트의 단어를 찾는다.
        sortting_list = []
        for text in request_data_arr:
            predict = model.predict(text) #모델 예측 시작
            classification = predict[0][0].split("__label__") #예측된 데이터에서 앞에 __label__ 제거

            #예측된 데이터의 라벨과 퍼센트 array형태로 만든다.
            sortting_obj = []
            sortting_obj.append(classification[1])
            sortting_obj.append(predict[1][0])
            sortting_list.append(sortting_obj)


        #리스트 순서 내림차순으로 솔팅
        sortting_list.sort(key=lambda x: -x[1])
        print(" ")
        print(sortting_list)
        print(" ")

        return sortting_list[0]

    #db에 있는 문서를 가져와서 보여주는 기능
    async def get_chatting_document(self, predict_result, session: Session = Depends(db.session)):

        # 예측 결과로 분류 값 추출
        # classification = predict_result[0][0].split("__label__")
        # 분류 카테고리로 db의 데이터 가져옴
        document = await self.get_one_chetting_condition(predict_result[0], session)

        if not document:
            result_json = {
                "classification": "분류되지 않음",
                "probability_predicted": "0.0",  # 예측 확률
                "document": "질문을 잘 이해하지 못했습니다.",
            }
        else:
            result_json = {
                "classification": document[0].categorey,
                "probability_predicted": predict_result[1],  # 예측 확률
                "document": document[0].document,
            }

        return result_json

    async def get_one_chetting_condition(
            #모프 종류, 추천 모프 데이터 가져오기
            self,
            classification,
            session: Session = Depends(db.session)):

        chattingBotSchema_datas = session.query(ChattingBotSchema).filter(ChattingBotSchema.categorey == str(text(classification))).all()
        return chattingBotSchema_datas