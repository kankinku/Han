DB = []
i=0

for stauts in range(1,10,1):
    str_user_input = input("\n키워드를 통해서 정보를 조작합니다. \nI : 새로운 정보 입력 \nP : 이전 정보 조회\nC : 현재 정보 조회\nN : 다음 정보 조회 \nU : 현재 정보 수정 \nD : 현재 정보 삭제 \nQ : 프로그램 삭제\n").upper()
    if str_user_input == 'Q':
        print("종료합니다.")
        status = 10
    elif str_user_input == 'I':
        DB.append(input("추가할 내용을 입력하세요. : "))
        i += 1
    elif str_user_input == 'P':
        if i < 2 :
            print("이전 정보가 없습니다.")
        else:
            print(f"이전 정보를 조회합니다. {DB[i-2]}")
    elif str_user_input == 'C':
        print(f"현재 정보를 조회합니다. : {DB[i-1]}")
    elif str_user_input == 'N':
        print("다음 정보 조회")
    elif str_user_input == 'U':
        print("현재 정보 수정")
        DB[i] = input("수정할 정보를 입력하세요. : ")
    elif str_user_input == 'D':
        try:
            print("현재의 정보를 삭제합니다.")        
            del(DB[i])
        except:
            print("삭제할 정보가 없습니다.")
    else:
        print("다시 입력해주세요. : ")