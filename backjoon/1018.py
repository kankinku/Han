M, N = map(int, input().split())
board = [list(input()) for _ in range(M)]

min_count = 64  # 최대 바꿔야 할 칸 수

for x in range(M - 7):
    for y in range(N - 7):
        count1 = 0  # W로 시작
        count2 = 0  # B로 시작
        for i in range(8):
            for j in range(8):
                current = board[x+i][y+j]
                if (i + j) % 2 == 0:
                    if current != 'W':
                        count1 += 1
                    if current != 'B':
                        count2 += 1
                else:
                    if current != 'B':
                        count1 += 1
                    if current != 'W':
                        count2 += 1
        min_count = min(min_count, count1, count2)

print(min_count)
