import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#\\KMEANS\\
df = pd.read_csv(r"C:\Users\jheas\Documents\PROJECTS (college)\AML\Mall Customers.csv")
print("Shape of data:", df.shape)
print(df.head())


plt.scatter(df['Age'], df['Spending Score (1-100)'])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Age vs Spending Score")
plt.show()

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending Score")
plt.show()

df.groupby('Gender')['Spending Score (1-100)'].mean().plot(kind='bar')
plt.title("Gender vs Spending Score")
plt.show()


#Selection of feature
from sklearn.cluster import KMeans
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
#Elbow Method
wcss = []
for i in range(1, 11):
   km = KMeans(n_clusters=i, random_state=42, n_init=10)
   km.fit(X)
   wcss.append(km.inertia_)


plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


#Applying KMeans
km = KMeans(n_clusters=5)
y_means = km.fit_predict(X)
print("Cluster labels for each customer:\n", y_means)
#Analysis of Cluster 3
print("Income of Cluster 3 customers:", X[y_means == 3, 0])


plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='blue', label='Cluster 0')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='red', label='Cluster 1')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='green', label='Cluster 2')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow', label='Cluster 3')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], color='pink', label='Cluster 4')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.legend()
plt.show()




#//Decision Tree//
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df['Customer Type'] = y_means
Y = df['Customer Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=42)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
classifier_pred = classifier.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test, classifier_pred))
print("Confusion Matrix:",confusion_matrix(y_test,classifier_pred))
print("Classification report:\n",classification_report(y_test,classifier_pred))






#//RANDOM FOREST//
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",accuracy_score(y_test, rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix:\n")
print(cm_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot(cmap='viridis')
plt.title("Random Forest Confusion Matrix")
plt.show()




#//LINEAR REGRESSION//
x = df['Annual Income (k$)'].values
y = df['Spending Score (1-100)'].values
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0
den = 0
for i in range(n):
    num += (x[i] - x_mean) * (y[i] - y_mean)
    den += (x[i] - x_mean) ** 2
m = num / den
c = y_mean - (m * x_mean)
print("\nSlope (m):", m)
print("Intercept (c):", c)

y_pred = []

for i in x:
    y_pred.append(m * i + c)

#R2 SCORE
ss_total = 0
ss_residual = 0

for i in range(n):
    ss_total += (y[i] - y_mean) ** 2
    ss_residual += (y[i] - y_pred[i]) ** 2

r2 = 1 - (ss_residual / ss_total)

print("R2 Score:", r2)

#//NEew customer prediction//


print("\nNEW CUSTOMER PREDICTION")

customer_id = int(input("Enter Customer ID: "))
gender = input("Enter Gender (Male/Female): ")
age = int(input("Enter Age: "))
income = float(input("Enter Annual Income (k$): "))

predicted_spending = m * income + c

print("\nPredicted Spending Score:",round(predicted_spending, 2))

# Predict customer cluster using KMeans
new_customer = np.array([[income, predicted_spending]])

cluster_prediction = km.predict(new_customer)

print("Predicted Customer Cluster:",cluster_prediction[0])

#Predicting customer type using Decision Tree
tree_prediction = classifier.predict(new_customer)

print("Decision Tree Predicted Customer Type:",tree_prediction[0])

#Predicting customer type using Random Forest
rf_prediction = rf.predict(new_customer)

print("Random Forest Predicted Customer Type:", rf_prediction[0])



# =========================================
# BUSINESS USE
# =========================================
# Premium customers:
# -> Luxury ads
# -> VIP membership
#
# Budget customers:
# -> Discounts and coupons
#
# Young customers:
# -> Fashion and gaming ads
#
# High predicted spending:
# -> Personalized recommendations
#
# This helps increase:
# -> Sales
# -> Customer retention
# -> Advertisement efficiency
# -> Mall profit

