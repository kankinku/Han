N = int(input())
p = 0

while N >= 0:
    if N % 5 == 0:
        print(p + (N // 5))
        break
    N -= 3
    p += 1
else:
    print(-1)
