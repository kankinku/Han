def info_insert():
    while True:
        name = input("이름을 입력하세요.")
        if len(name) < 2 or len(name) > 4:
            print("이름은 2글자 이상 4글자 이하로 입력해주세요.")
        else:
            break
    while True:
        gender = input("성별을 입력하세요. (M/F)").upper()
        if len(gender) != 1:
            print("성별은 1글자만 입력해주세요.")
        else:
            break
    while True:
        email = input("이메일 주소를 입력하세요.")
        if "@" not in email:
            print("이메일 주소는 @를 포함해야 합니다.")
        else:
            break
    while True:
        year = input("출생년도를 입력하세요.")
        if not year.isdigit():
            print("정수를 입력해주세요.")
        else:
            year = int(year)
            if year < 1900 or year > 2024:
                print("출생년도는 1900에서 2024 사이로 입력해주세요.")
            else:
                break
            
    cust_list.append({'이름': name, '성별': gender, '이메일': email, '출생년도': year})

def view_nowInfo(now_index):
    if len(cust_list) != 0:
        print("현재 정보 조회")
        print(cust_list[now_index])
    else:
        print("입력된 정보가 없습니다.")

def view_beforeInfo(now_index):
    print("이전 정보 조회")
    if now_index > 0:
        now_index -= 1
        print(cust_list[now_index])
    else:
        print("이전의 데이터가 없습니다.")
    return now_index

def view_nextInfo(now_index):
    print("다음 정보 조회")
    if now_index < len(cust_list) - 1:
        now_index += 1
        print(cust_list[now_index])
    else:
        print("다음 데이터가 없습니다.")
    return now_index

def fix_nowInfo(now_index):
    print("현재 정보 수정")
    if len(cust_list) == 0:
        print("수정할 정보가 없습니다.")   
    if input("이름을 수정하시겠습니까? (Y/N)").upper == "Y":
        name = input(f"이름을 입력하세요. ({cust_list[now_index]['이름']}): ")
    if input("성별을 수정하시겠습니까? (Y/N)").upper == "Y":
        gender = input(f"성별을 입력하세요. (M/F) ({cust_list[now_index]['성별']}): ").upper()
    if input("이메일을 수정하시겠습니까? (Y/N)").upper == "Y":
        email = input(f"이메일 주소를 입력하세요. ({cust_list[now_index]['이메일']}): ")
    if input("출생년도를 수정하시겠습니까? (Y/N)").upper == "Y":
        year = input(f"출생년도를 입력하세요. ({cust_list[now_index]['출생년도']}): ")

    if name:
        cust_list[now_index]['이름'] = name
    if gender:
        cust_list[now_index]['성별'] = gender
    if email:
        cust_list[now_index]['이메일'] = email
    if year:
        cust_list[now_index]['출생년도'] = int(year) if year.isdigit() else cust_list[current_index]['출생년도']

def Delete_nowInfo(now_index):
    print("현재 정보 삭제")
    if len(cust_list) > 0:
        cust_list.pop(now_index)
        if now_index >= len(cust_list) and len(cust_list) > 0:
            now_index = len(cust_list) - 1
    else:
        print("삭제할 정보가 없습니다.")


cust_list = [
    {'이름': 'han', '성별': 'M', '이메일': 'wlsdn1232580@gmail.com', '출생년도': '2001'},
    {'이름': 'jin', '성별': 'M', '이메일': 'hanjinu123@gmail.com', '출생년도': '2002'}
]

now_index = -1

while True:
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
        info_insert()
    elif menu_choice == "C":  
        view_nowInfo(now_index)
    elif menu_choice == "P":      
        now_index = view_beforeInfo(now_index)
    elif menu_choice == "N":  
        now_index = view_nextInfo(now_index)
    elif menu_choice == "U":  
        fix_nowInfo(now_index)
    elif menu_choice == "D":  
        Delete_nowInfo(now_index)
    elif menu_choice == "Q":
        print("프로그램을 종료합니다.")
        break

    else:
        print("입력 하신 메뉴가 없습니다. \n다시 입력하세요.")