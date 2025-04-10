n = int(input())
count = 0
result = 666
i = 0

while True:
    if str(result) in str(i):
        count += 1
    
    if count == n:
        break

    i += 1

print(i)