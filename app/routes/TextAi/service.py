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

base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
import fasttext

class text_ai_service:

    async def transfrom_request_question(self, request_data):

        trans_data = request_data

        print("초기 데이터*****")
        print(trans_data)
        print("초기 데이터*****")

        #한글만 남기고 다 지울것
        trans_data = re.sub(r"[^\uAC00-\uD7A3\s]", "", trans_data)

        #의가 있는 경우
        if trans_data.find('의') > -1:
            trans_data = trans_data.replace('의', '')

        #크레스티드가 없는 경우
        #크레스티드가 없는 경우
        if trans_data.find('크레스티드') == -1:
            if trans_data.find('크레') > -1:
                trans_data = trans_data.replace('크레', '크레스티드 게코')
                print("크레 찾음")

        print("변환 데이터*****")
        # print(trans_data.find('크레'))
        print(trans_data)
        print("변환 데이터*****")

        return trans_data

    #챗봇 답변 주는 기능
    async def response_chatting_bot(self, request_data):
        model = fasttext.load_model(base_dir + "/core/chatting_bot_model/model_cooking_reptile.bin")

        predict = model.predict(request_data)
        return predict

    #db에 있는 문서를 가져와서 보여주는 기능
    async def get_chatting_document(self, predict_result, session: Session = Depends(db.session)):

        # 예측 결과로 분류 값 추출
        classification = predict_result[0][0].split("__label__")
        # 분류 카테고리로 db의 데이터 가져옴
        document = await self.get_one_chetting_condition(classification[1], session)

        if not document:
            result_json = {
                "classification": "분류되지 않음",
                "probability_predicted": "0.0",  # 예측 확률
                "document": "질문을 잘 이해하지 못했습니다.",
            }
        else:
            result_json = {
                "classification": document[0].categorey,
                "probability_predicted": predict_result[1][0],  # 예측 확률
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