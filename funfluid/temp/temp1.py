import os

path = '/c/d/a1.txt'
a, b = os.path.splitext(path)
print(a)
print(os.path.splitext(a))
