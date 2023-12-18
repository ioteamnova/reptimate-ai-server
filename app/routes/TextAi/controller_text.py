from typing import List
from fastapi import Depends, APIRouter
from os import path
from sqlalchemy.orm import Session
from routes.TextAi.dtos.ChattingBot_dto import ChattingBot
from core.database.conn import db
from routes.TextAi.service import text_ai_service


base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
router = APIRouter(prefix='/text_ai')

@router.post("/chatting_bot", summary="채팅 봇", description="자연어 생성 모델을 통해 답변을 추출한다.")
async def chattingBot(data: ChattingBot = Depends(),
                      text_ai_service: text_ai_service = Depends(text_ai_service),
                      session: Session = Depends(db.session)):

    question = await text_ai_service.transfrom_request_question(data.request_text)

    # print("question")
    # print(question)
    # print(len(question))
    # print('question')

    if len(question) != 0:
        # 챗봇 분류해주는 기능
        predict_result = await text_ai_service.response_chatting_bot(question)

        # print("predict_result")
        # print(predict_result)
        # print('predict_result')

        # db에서 분류 class에 맞는 내용 가져오는 기능
        document = await text_ai_service.get_chatting_document(predict_result, session)

        return document
    else:
        return "질문을 잘 이해하지 못했어요."

