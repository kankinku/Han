N,K = map(int, input().split())
score=list(map(int,input().split()))
score.sort(reverse=True)
for _ in range(K-1):
    del score[0]
print(score[0])
