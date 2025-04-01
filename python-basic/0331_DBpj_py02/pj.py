# 고객 정보 리스트
cust_list = [['han','M','wlsdn1232580@gmail.com','2001'],
            ['jin','M','hanjinu123@gmail.com','2002']]
# 현재 위치 정보 변수
current_index = len(cust_list)-1

while True:
    print(current_index)
    menu_choice = input(
        """
            I - 고객 정보 입력
            C - 현재 고객 정보 조회
            P - 이전 고객 정보 조회
            N - 다음 고객 정보 조회
            U - 현재 고객 정보 수정
            D - 현재 고객 정보 삭제
            Q - 프로그램 종료\n"""
    ).upper()

    if menu_choice == "I":
        name = input("이름을 입력하세요.")
        gender = input("성별을 입력하세요. (M/F)").upper()
        email = input("이메일 주소를 입력하세요.")
        year = int(input("출생년도를 입력하세요."))

        cust_list.append([name, gender, email, year])
        current_index = len(cust_list) - 1

    elif menu_choice == "C":
        if len(cust_list) != 0:
            print("현재 정보 조회")
            print(cust_list[current_index])
        else:
            print("입력된 정보가 없습니다.")

    elif menu_choice == "P":
        print("이전 정보 조회")
        if current_index > 0:
            current_index -= 1
            print(cust_list[current_index])
        else:
            print("이전의 데이터가 없습니다.")
        pass

    elif menu_choice == "N":
        print("다음 정보 조회")
        if current_index < len(cust_list)-1:
            current_index += 1
            print(cust_list[current_index])
        else:
            print("다음 데이터가 없습니다.")
        pass

    elif menu_choice == "U":
        print("현재 정보 수정")
        name = input("이름을 입력하세요. ({})".format(cust_list[current_index][0]))
        gender = input("성별을 입력하세요. (M/F)").upper()
        email = input("이메일 주소를 입력하세요.")
        year = int(input("출생년도를 입력하세요."))

        cust_list[current_index] = [name, gender, email, year]

    elif menu_choice == "D":
        print("현재 정보 삭제")
        cust_list.pop(current_index)

    elif menu_choice == "Q":
        print("프로그램을 종료합니다.")
        break

    else:
        print("입력 하신 메뉴가 없습니다. \n다시 입력하세요.")
