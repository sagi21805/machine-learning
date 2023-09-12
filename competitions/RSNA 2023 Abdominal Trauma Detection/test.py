import os
import platform
dict1 = {
    1: 10, 
    2: 20, 
    3: 30
}

dict2 = {
    1: 5, 
    2: 10, 
    3: 15
}

for key, val1, val2 in zip(dict1, dict1.values(), dict2.values()):
    dict1[key] = (val1, val2)    
    
print(dict1)
list = [1, 2, 3]
list2 = [4, 5, 6]
print(list + list2)