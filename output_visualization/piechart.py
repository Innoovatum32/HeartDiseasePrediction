import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
# Data to pie Chart
labels = 'Naive Bayes','K-Nearest Neighbors','Kernelized SVM','Decision Tree','Logistic Regression'
sizes = [180,130,170,155,165]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','darkblue']
explode = (0.1, 0, 0, 0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

#nb:0.86-180
#knn:0.78-130
#svm:0.852-170
#dt:0.81-155
#lg:0.85-165

#barchart
objects = ('NB','K-NN','SVM','DT','LR')
y_pos = np.arange(len(objects))
performance = [8.6,7.8,8.5,8.1,8.5]

plt.bar(y_pos, performance, align='center', alpha=0.6)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Comparing Different ML models for heart disease prediction')

plt.show()