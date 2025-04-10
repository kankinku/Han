import sys

T = int(sys.stdin.readline())
N = []
for _ in range(T):
    N.append(int(sys.stdin.readline().strip()))

for i in sorted(N):
    print(i)

