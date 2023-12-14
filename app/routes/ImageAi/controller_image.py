from typing import List
from fastapi import Depends, UploadFile, APIRouter, File
from os import path
from sqlalchemy.orm import Session
from routes.ImageAi.dtos.ValueAnalyzer_dto import ValueAnalyzerCreate,ValueAnalyze
from core.database.conn import db
from routes.ImageAi.service import image_ai_service
from utils.FileChecker import FileChecker


base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
router = APIRouter(prefix='/image_ai')

@router.post("/value_analyzer", summary="도마뱀 가치 판단 기능", description="*files의 첫 번째에는 Top 이미지 두번쨰에는 left 마지막은 right")
async def assessValue(data: ValueAnalyze = Depends(), files: List[UploadFile] = File(...),
        image_ai_service: image_ai_service = Depends(image_ai_service),
        session: Session = Depends(db.session)):
    #이미지 유효성 검사
    await FileChecker.imgCheck(files)
    # 가치 판단 기능 실행
    result = await image_ai_service.assess_value(data, files)  # assess_value 메서드 호출
    print("result1: ", result)
    # 결과 데이터 및 이미지 s3 저장
    result2 = await image_ai_service.analyzer_auto_save(result, files, session)

    return result2

@router.post("/value_analyzer_nosave", summary="도마뱀 가치 판단 기능 (저장 안되는 기능)", description="*files의 첫 번째에는 Top 이미지 두번쨰에는 left 마지막은 right")
async def assessValue_nosave(data: ValueAnalyze = Depends(), files: List[UploadFile] = File(...),
        image_ai_service: image_ai_service = Depends(image_ai_service),
        session: Session = Depends(db.session)):
    #이미지 유효성 검사
    await FileChecker.imgCheck(files)
    # 가치 판단 기능 실행
    result = await image_ai_service.assess_value(data, files)  # assess_value 메서드 호출
    print("result1: ", result)

    # 결과 저장하지 않고 정제
    result2 = await image_ai_service.analyzer_refine_data(result, files)

    return result2

@router.post("/analyzer_save", summary="가치 판단 후 결과 저장하는 기능", description="*로그인 되어야 저장 가능합니다. 로그인 안됬으면 로그인 후에 해당 기능 실행해주세요!")
async def analyzer_save(
        idx: int,
        userIdx: int,
        petName: str,
        image_ai_service: image_ai_service = Depends(image_ai_service),
        session: Session = Depends(db.session)):

    # 로그인이 되어 있는 상태에서 가치 판단 결과 저장
    result = await image_ai_service.analyzer_save(idx, userIdx, petName, session)  # assess_value 메서드 호출

    return result

@router.post("/gender_discrimination", summary="암수 구분 기능")
async def gender_discrimination(
        file: UploadFile,
        image_ai_service: image_ai_service = Depends(image_ai_service)):
    # 이미지 유효성 검사
    await FileChecker.imgCheck(file)
    genderResult = await image_ai_service.gender_discrimination(file)
    return genderResult


@router.post("/linebreeding_recommend", summary="도마뱀 라인브리딩 추천", description="*files의 첫 번째에는 Top 이미지 두번쨰에는 Left 마지막은 Right")
async def linebreedingRecommend(data: ValueAnalyze = Depends(), files: List[UploadFile] = File(...),
        image_ai_service: image_ai_service = Depends(image_ai_service),
        session: Session = Depends(db.session)):

    # 가치 판단 기능 실행
    UserResult = await image_ai_service.assess_value(data, files)  # assess_value 메서드 호출
    # 결과 데이터 및 이미지 s3 저장
    # await image_ai_service.analyzer_auto_save(UserResult, files, session)

    print("UserResult")
    print(UserResult)
    print("UserResult")
    get_analyzer_result = await image_ai_service.get_analyzer_data(UserResult, session)  # 분석

    return get_analyzer_result