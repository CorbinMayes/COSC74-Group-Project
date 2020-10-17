import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")

# opening the file, 
main_list = []
for line in open('/Users/admin/COSC74 files/sample_data.json', 'r'):
    main_list.append(json.loads(line))
    
def is_verified(my_list):
    
    x_list = []
    for i in my_list:
        if i['verified'] == False:
            x_list.append(1)
        else:
            x_list.append(0)

    return x_list

def find_reviewText_lengths(my_list):
    
    list_of_lengths = []
    
    for i in my_list:
        string = i['reviewText']
        list_of_lengths.append(len(string.split()))
        
    return list_of_lengths

def build_y(my_list):
    y_list = []
    x_list = is_verified(main_list)
    for i in my_list:
        if i['overall'] < 4.4:
            y_list.append(1)
        else:
            y_list.append(0)
            
def build_array(my_list_1, my_list_2):
    
    array_list = []
    
    for i in range(0, len(my_list_1)):
        nested_list = []
        nested_list.append(my_list_1[i])
        nested_list.append(my_list_2[i])
        array_list.append(nested_list)
        
    return array_list

verified_set = is_verified(main_list)
reviewText_set = find_reviewText_lengths(main_list)
y_set = build_y(main_list)

compiled_list = build_array(reviewText_set, verified_set)
X = np.array(compiled_list)

Y = y_set

clf = svm.SVC(kernel = 'linear', C = 1.0)
clf.fit(X,y)

w = clf.coef_[0]


a = -w[0] / w[1]

xx = np.linspace(0,180)
yy = a * xx - clf.intercept_[0]/ w[1]

h0 = plt.plot(xx, yy, 'k-', label = "something")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()


"""1 means that the review is not awesome, while 0 means that the algorithm predicted awesome."""
print(clf.predict([[40, 0]]))
print(clf.predict([[60, 1]]))
print(clf.predict([[20, 0]]))
print(clf.predict([[90, 1]]))
print(clf.predict([[139, 0]]))
print(clf.predict([[10, 1]]))