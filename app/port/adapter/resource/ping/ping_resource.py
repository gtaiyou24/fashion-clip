from fastapi import APIRouter


router = APIRouter(
    prefix='/ping',
    tags=['ヘルスチェック']
)


@router.get('', name='ヘルスチェック用のエンドポイント')
def check() -> str:
    return "OK"
