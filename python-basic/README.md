현재 문제점 :
python 명령어가 사라짐
pip는 사용 불가능하고
py -m pip ~를 사용해야 함

-> 해결 (python 폴더를 PATH 환경변수에 추가하고, pip도 동일하게 설정한다.)

source .venv/Scripts/activate : 가상환경에 들어가는 명령어
deactivate : 가상환경에서 나오는 명령어

------------

uv venv : 가상환경을 생성하여 환경변수의 변경 없이 다른 Python 버전을 사용할 수 있도록 만들어준다.

source .venv/Scripts/activate : 가상환경에 들어가는 명령어
deactivate : 가상환경에서 나오는 명령어

------------

uv sync
1. `pyproject.toml`, `requirements.txt` : 기본 설정이 저장된 파일이다.

-------------

alt + shift + 화살표 = 라인 복사

숫자 옆을 눌러서 빨간색 표시를 만들면, 그 지점까지 디버깅함

