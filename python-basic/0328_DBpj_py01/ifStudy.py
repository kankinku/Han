# my = input("너의 정체가 무엇이냐?\n")
# if my == '가수':
#     print("그럼 노래를 불러라")
# else:
#     print(f"{my}이라니, 그럼 꺼져라")

store = list(map(str,input("오픈 여부를 입력하세요.(H = 햄버거,S = 샌드위치 ,X = 닫음)\n").split()))
if 'H' in store:
    print("햄버거 가게는 열었습니다.")
if 'S' in store:
    print("샌드위치 가게는 열었습니다.")
if store == 'X':
    print("열린 가게가 없다.")

