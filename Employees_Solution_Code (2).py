import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff


# Importing dataset and examining it
dataset = pd.read_csv("Employees.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
dataset['BusinessTravel'] = dataset['BusinessTravel'].map({'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2})
dataset['Department'] = dataset['Department'].map({'Human Resources':0, 'Research & Development':1, 'Sales':2})
dataset['EducationField'] = dataset['EducationField'].map({'Medical':0, 'Human Resources':1, 'Life Sciences':2, 'Technical Degree':3, 'Marketing':4, 'Other': 5})
dataset['JobRole'] = dataset['JobRole'].map({'Healthcare Representative':0, 'Human Resources':1, 'Laboratory Technician':2, 'Manager':3, 'Manufacturing Director':4, 'Research Director': 5, 'Research Scientist':6, 'Sales Executive':7, 'Sales Representative':8})
dataset['MaritalStatus'] = dataset['MaritalStatus'].map({'Divorced':0, 'Married':1, 'Single':2})
dataset['Gender'] = dataset['Gender'].map({'Male':0, 'Female':1})
dataset['OverTime'] = dataset['OverTime'].map({'No':0, 'Yes':1})

print(dataset.info())
# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')

# Dropping columns with high correlation + causation
X = dataset.drop(['YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'PercentSalaryHike', 'JobLevel', 'JobRole'], axis = 1)
print(type(X))
print(X.shape)

# Dividing data into subsets
#Personal Data
subset1 = X[['Age','Gender','MaritalStatus','Education', 'EducationField', 'DistanceFromHome' ]]

#Work Data
subset2 = X[['Department','BusinessTravel','OverTime','StockOptionLevel','TrainingTimesLastYear','YearsSinceLastPromotion','PerformanceRating']]

#Life Quality Data
subset3 = X[['JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement', 'WorkLifeBalance']]

#Churn factors
subset4 = X[['JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement', 'WorkLifeBalance','OverTime', 'StockOptionLevel', 'YearsSinceLastPromotion', 'PerformanceRating']]

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)
X4 = feature_scaler.fit_transform(subset4)

# Analysis on subset1 - Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

age = list(X['Age'])
gender = list(X['Gender'])
marital = list(X['MaritalStatus'])
education = list(X['Education'])
educationfield = list(X['EducationField'])
distance = list(X['DistanceFromHome'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Age: {a}; Gender: {b}; MaritalStatus:{c}, Education:{d}, EducationField:{e}, DistanceFromHome:{f}' for a,b,c,d,e,f in list(zip(age,gender,marital,education,educationfield,distance))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')

# Analysis on subset2 - Work Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X2)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

dep = list(X['Department'])
btravel = list(X['BusinessTravel'])
ot = list(X['OverTime'])
solevel = list(X['StockOptionLevel'])
ttly = list(X['TrainingTimesLastYear'])
yslp = list(X['YearsSinceLastPromotion'])
prating = list(X['PerformanceRating'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'dep: {a}; btravel: {b}; ot:{c}, solevel:{d}, ttly:{e}, yslp:{f}, prating:{g}' for a,b,c,d,e,f,g in list(zip(dep,btravel,ot,solevel,ttly,yslp,prating))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE2.html')

# Analysis on subset3 - Life Quality Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X3)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X3)

jsatifaction = list(X['JobSatisfaction'])
esatisfaction = list(X['EnvironmentSatisfaction'])
ji = list(X['JobInvolvement'])
wlb = list(X['WorkLifeBalance'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'jsatisfaction: {a}; esatisfaction: {b}; ji:{c}, wlb:{d}' for a,b,c,d in list(zip(jsatifaction,esatisfaction,ji,wlb))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE3.html')

# Analysis on subset4 - Churn Factos
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X4)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X4)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X4)

jsatisfaction = list(X['JobSatisfaction'])
esatisfaction = list(X['EnvironmentSatisfaction'])
ji = list(X['JobInvolvement'])
wlb = list(X['WorkLifeBalance'])
ot = list(X['OverTime'])
solevel = list(X['StockOptionLevel'])
yslp = list(X['YearsSinceLastPromotion'])
prating = list(X['PerformanceRating'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'jsatisfaction: {a}; esatisfaction: {b}; ji:{c}, wlb:{d}, ot:{e}, solevel:{f}, yslp:{g}, prating:{h}' for a,b,c,d,e,f,g,h in list(zip(jsatisfaction,esatisfaction,ji,wlb,ot,solevel,yslp,prating))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE4.html')
