
# coding: utf-8

# # Логистическая регрессия с подбором параметра С при кросс валидации

# In[443]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import rcParams
import mglearn
from sklearn.metrics import roc_auc_score, roc_curve


# In[401]:


cancer = load_breast_cancer()


# In[402]:


#Разбивка данных на обчающий, проверочный и тестовый наборы
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, 
                                                         stratify = cancer.target, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval,y_trainval, random_state = 0)


# In[404]:


# Рассмотрим параметр C на интервале от 0 до 100
param = np.arange(0.01,100,0.5)
scores = []
for c in param:
    log_reg = LogisticRegression(C = c)
    scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
    scores.append(np.mean(scor))

plt.plot(param, scores,label = 'cross validation')
plt.ylabel("Правильность")
plt.xlabel("C")
plt.legend()
plt.plot()
rcParams['figure.figsize'] = 10, 5


# In[405]:


# Рассмотрим участок до 20 
n = np.arange(-10,20,1, dtype=float)
param = []
for i in n:
    param.append(10**i)
scores = []
for c in param:
    log_reg = LogisticRegression(C = c)
    scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
    scores.append(np.mean(scor))

plt.plot(np.log10(param), scores,label = 'cross validation')
plt.ylabel("Правильность")
plt.xlabel("C")
plt.legend()
plt.plot()
rcParams['figure.figsize'] = 10, 5

print(param)

# видим, что график колеблется и достигает высоких значениях в местах около точек 5,7 и 13


# In[406]:


# расммотрим этот участок
n = np.arange(0,12,0.5, dtype=float)
param = []
for i in n:
    param.append(10**i)
scores = []
for c in param:
    log_reg = LogisticRegression(C = c)
    scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
    scores.append(np.mean(scor))

plt.plot(np.log10(param), scores,label = 'cross validation')
plt.ylabel("Правильность")
plt.xlabel("C")
plt.legend()
plt.plot()
rcParams['figure.figsize'] = 10, 5


# In[407]:


# Рассмотрим 3 участка

fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # задаем сетку для графиков

sector_1 = np.arange(0.001,2.3,0.1, dtype=float) # задаем участки
sector_2 = np.arange(4,6,0.1, dtype=float)
sector_3 = np.arange(8, 10, 0.1, dtype=float)


for sector,ax in zip([sector_1,sector_2,sector_3], axes):
    param = []
    for point in sector:
        param.append(10**point)
        scores = []
        for c in param:
            log_reg = LogisticRegression(C = c)
            scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
            scores.append(np.mean(scor))
    ax.set_title("C interval [{} - {}]".format(sector[1],round(sector[-1],1)))    
    ax.plot(np.log10(param), scores,label = 'cross validation')
    ax.set_ylabel("Правильность")
    ax.set_xlabel("C")
    ax.legend()
    ax.plot()
    
    max_score = 0                        # определяем максимальное значение правильности на заданном интервале
    for item in scores:
        if item > max_score:
            max_score = item
    
    max_C = []                           # при каком значении С достигается max значение
    for k in range(len(scores)):
        if scores[k] == max_score:
            max_C.append(np.log10(param)[k])
            
    print ('Промежуток [{}-{}]'.format(sector[1],round(sector[-1],1)))
    print('Наибольшее значение правильности, равное {}'.format(max_score))
    print('при С = {}\n'.format(max_C))
   
    #rcParams['figure.figsize'] = 20, 15            


# In[408]:


# sector_1 max C = 1.2 
# sector_2 max C = 4.6 
# сравним

log_reg = LogisticRegression(C = 1.2)
scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
print('Приавильность при С = 1.2: {}'.format(np.mean(scor)))

log_reg = LogisticRegression(C = 4.6)
scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
print('Приавильность при С = 4.6: {}'.format(np.mean(scor)))


# In[409]:


# при C = 4.6 достигается наибольшая точность 

# влияние кол-ва блоков в кросс валидации на ззначение правильности
blocks = np.arange(2,22,1)
scores = []
for b in blocks:
    log_reg = LogisticRegression(C = 4.6)
    scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = b)
    scores.append(np.mean(scor))

plt.plot(blocks, scores,label = 'cross validation')
plt.ylabel("Правильность")
plt.xlabel("Кол-во блоков")
plt.legend()
plt.plot()
rcParams['figure.figsize'] = 5, 5


# In[410]:


# При параметре C = 4.6 и cv = 10 достигается правильность больше 96%

log_reg = LogisticRegression(C = 4.6)
scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 10)
print(np.mean(scor))


# In[412]:


log_reg.fit(X_trainval,y_trainval)
print("Правильность на обучающем наборе: {:.2f}".format(log_reg.score(X_trainval, y_trainval)))
print("Правильность на тестовом наборе: {:.2f}".format(log_reg.score(X_test, y_test)))


# In[413]:


plt.figure()

C = [1.2,4.6]
for c in C:
    log_reg = LogisticRegression(C = c).fit(X_trainval, y_trainval)
    accuracy = log_reg.score(X_test, y_test)
    auc = roc_auc_score(y_test, log_reg.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test , log_reg.decision_function(X_test))
    print("C = {:.2f} правильность = {:.4f} AUC = {:.3f}".format(c, accuracy, auc))
    plt.plot(fpr, tpr, label="C={:.3f}".format(c))
    
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.show()


# ## Установка порогового значения по вероятности

# ### Используем для порогового значения тренировочный набор

# In[414]:


X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, 
                                                         stratify = cancer.target, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval,y_trainval, random_state = 0)

# изначально порог на decision_function = 0, redict_proba = 0.5

log_reg = LogisticRegression(C = 4.6) 
log_reg.fit(X_train, y_train) 

precision, recall, tresholds = precision_recall_curve(y_valid, log_reg.predict_proba(X_valid)[:,1])
close_default = np.argmin(np.abs(tresholds - 0.5))
plt.plot(precision[close_default], recall[close_default], '^', c='k',
         markersize=7, label="порог вероятности 0.5", fillstyle="none", mew=2)

plt.plot(precision, recall, label = 'кривая точности-полноты')
plt.xlabel('точность')
plt.ylabel('полнота')
plt.legend(loc = 'best')
plt.show()

rcParams['figure.figsize'] = 5, 5


print(classification_report(y_valid,log_reg.predict(X_valid))) 


# In[415]:


# при такой же полноте, можем увеличить показатель точноси до 97%, увеличичив порог вероятности до 70%
# (только там, где алгоритм сильнее уверен, он должен ставить метку класса 1. Порог вероятности д б больше)


precision, recall, tresholds = precision_recall_curve(y_valid, log_reg.predict_proba(X_valid)[:,1])

for i in np.arange(0.1,1,0.2):
    close_point = np.argmin(np.abs(tresholds - i))
    
    
    plt.plot(precision[close_point], recall[close_point], 'x',
             markersize=12, label="порог вероятности {:.2f}".format(i), fillstyle="none", mew=2) 
    
plt.plot(precision, recall, label = 'кривая точности-полноты')
plt.xlabel('точность')
plt.ylabel('полнота')
plt.title('Кривая точности-полноты для Logistic Regression (C = 4.6) при разных порогах уверенности модели')
plt.legend(loc = 'best')
plt.show()

rcParams['figure.figsize'] = 5, 5

print('tresholds = 0.5')
print(classification_report(y_valid,log_reg.predict(X_valid))) 
print()
print('tresholds = 0.7')
y_pred_rise_threshold = log_reg.predict_proba(X_valid) >= 0.7
print(classification_report(y_valid,y_pred_rise_threshold[:,1]))


# ###### при значении по умолчанию (0.5) модель дает достаточно хорошие показатели. Однако, при настройке их можно оптимизировать

# In[416]:


precision, recall, tresholds = precision_recall_curve(y_valid, log_reg.predict_proba(X_valid)[:,1])

probability = np.arange(0.1,1,0.2)
f_scores = []

for i in probability:

    #print('tresholds = {:.2f}'.format(i))
    y_pred_change_threshold = log_reg.predict_proba(X_valid) >= i
    f_scores.append(f1_score(y_valid,y_pred_change_threshold[:,1]))
    
    #print('f1 = {:2f}'.format(f1_score(y_valid,y_pred_change_threshold[:,1])))


plt.plot(probability, f_scores, label = 'кривая порог вероятности-f1')  
plt.xlabel('порог вероятности')
plt.ylabel('f1')
plt.title('Кривая зависимости f1 от порога вероятности для Logistic Regression (C = 4.6)')
plt.legend(loc = 'best')
plt.show()

rcParams['figure.figsize'] = 5, 5


# In[399]:


from sklearn.metrics import roc_curve
fpr, recall, tresholds = roc_curve(y_valid, log_reg.predict_proba(X_valid)[:,1])

probability = np.arange(0.1,1,0.2)
f_scores = []

for i in probability:
    close_point = np.argmin(np.abs(tresholds - i))
    plt.plot(fpr[close_point], recall[close_point], 'o', markersize=10,
             label="порог {:.2f}".format(i), fillstyle="none",mew=2)
    


plt.plot(fpr, recall, label="ROC-кривая")
plt.xlabel("FPR")
plt.ylabel("TPR (полнота)")
plt.title('ROC-кривая для Logistic Regression (C = 4.6) при разных порогах уверенности модели')
plt.legend(loc='best')
plt.show()


# #### На графике видно, что при пороге вероятности = 0.7 модель достигает наибольшей точности с нулевым кол-вом ложно-положитеьных результатов (т.е. нет меток, которые модель определила бы как 1, а она оказалась 0)

# ## Cross Validation

# In[417]:


from sklearn.metrics import  f1_score, precision_recall_curve


# In[418]:


cancer = load_breast_cancer()


# In[419]:


data = cancer.data
target = cancer.target


# In[420]:


print(data.shape)
print(target.shape)


# In[421]:


X_train_cross_val, X_test_cross_val, y_train_cross_val, y_test_cross_val = train_test_split(data,target, 
                                                                                            stratify = target, 
                                                                                            random_state = 0)


# In[422]:


print(X_train_cross_val.shape)
print(y_train_cross_val.shape)


# In[424]:


split_x = np.array_split(X_train_cross_val, 5)
split_y = np.array_split(y_train_cross_val, 5)

f1 = []
scores = []

for i in range(len(split)):
    
    x = split_x.copy()
    y = split_y.copy()
    
    block_x_test = x.pop(i)
    block_y_test = y.pop(i)
    #print(block_x_test.shape)
    #print(block_y_test.shape)
    
    block_x_train_list = x
    block_y_train_list = y
    
    block_x_train_array = np.array(block_x_train_list[0])
    block_y_train_array = np.array(block_y_train_list[0])
    
    for m,n in zip(block_x_train_list[1:],block_y_train_list[1:]):
        block_x_train_array = np.concatenate((block_x_train_array,m), axis = 0)
        block_y_train_array = np.concatenate((block_y_train_array,n), axis = 0)
        
    
    #print(block_x_train_array.shape)
    #print(block_y_train_array.shape)
    
    log_reg_cv = LogisticRegression(C = 4.6).fit(block_x_train_array,block_y_train_array)
    pred_logreg = log_reg_cv.predict(block_x_test)
    
    precision, recall, tresholds = precision_recall_curve(block_y_test, 
                                                          log_reg_cv.predict_proba(block_x_test)[:,1])
    #close_default = np.argmin(np.abs(tresholds - 0.5))
    #plt.plot(precision, recall, label="iter {}".format(i))


    
    scores.append(log_reg_cv.score(block_x_test,block_y_test))
    f1.append(f1_score(block_y_test,pred_logreg))
    
    print('Итерация №{}'.format(i))
    print('правильность {:.2f}'.format(log_reg_cv.score(block_x_test,block_y_test)))
    print('f1 мера {:.2f}'.format(f1_score(block_y_test,pred_logreg)))
    print()

print('средняя f1 мера по cross validation {:.2f}'.format(np.mean(f1)))
print('средняя правильность по cross validation {:.2f}'.format(np.mean(scores)))    


#plt.plot(precision, recall, label = 'кривая точности-полноты')
#plt.xlabel('точность')
#plt.ylabel('полнота')
#plt.legend(loc = 'best')
#plt.show()

#rcParams['figure.figsize'] = 5, 5


# ## Размер контрольного множества

# In[425]:


#Разбивка данных на обчающий и тестовый наборы с разной величиной контрольной выборки

def procent_of_test_size(control_sizes, data, target):
    scores = []
    data_strucrure = {'%_test_size': ['original'],  # формируем имена колонок
                      'train_size':[cancer.data.shape[0]],
                      'test_size': [cancer.data.shape[0]] , 
                      'mean': [round(np.mean(data))] , 
                      'var': [round(np.var(data))] ,
                      'score': [0]}
    df = pd.DataFrame(data_strucrure)        # создаем датафрейм 
      
    for size in control_sizes:
        train_size = 1 - size
        test_percent = size*100
        X_trainval, X_test, y_trainval, y_test = train_test_split(data, target, 
                                                                  stratify = target, 
                                                                  random_state = 0,
                                                                  test_size = size,
                                                                  train_size = train_size)
        train_size = X_trainval.shape 
        test_size = X_test.shape
        mean = np.mean(X_trainval)
        var = np.var(X_trainval)
    
        log_reg = LogisticRegression(C = 4.6)
        scor = cross_val_score(log_reg, X_trainval, y_trainval, cv = 5)
        score = np.mean(scor)
        scores.append(score)
    
    
        new_row = {'%_test_size': test_percent, 
                   'train_size': train_size[0], 
                   'test_size': test_size[0], 
                   'mean': round(mean,2), 
                   'var': round(var,2),
                   'score': round(score,3)}
        
        
        df = df.append(new_row, ignore_index=True)

        
    plt.plot(control_sizes*100, scores)
    plt.ylabel("Правильность")
    plt.xlabel("Размер контрольного множества, %")
    plt.title('Правильность логистической регрессии при разных размерах контрольной выборки')
    plt.plot()
    rcParams['figure.figsize'] = 5, 5        
    
    return df


# In[426]:


control_sizes = np.arange(0.1,1.0,0.05)
cancer_data_df = procent_of_test_size(control_sizes,cancer.data, cancer.target )


# In[427]:


cancer_data_df


# In[428]:


print('max mean value:',cancer_data_df["mean"].max())
print('min mean value:',cancer_data_df["mean"].min())

print()

print('max variance value:',cancer_data_df["var"].max())
print('min variance value:',cancer_data_df["var"].min())


# In[429]:


plt.hist(cancer_data_df["mean"][1:])
plt.title('Распределение средних значений')
plt.plot()


# # Отбор значимых признаков (PCA) и классификация

# ### PCA

# In[430]:


from sklearn.preprocessing import StandardScaler


# In[431]:


# разбиваем выборку на тренировочный и тестовый наборы 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                         stratify = cancer.target, random_state = 0)

# масштабируем данные, чтобы они имели var = 1
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[432]:


np.var(X_train_scaled)


# In[434]:


# применяем метод PCA (главныйх компонент), чтобы снизить размерность данных (с меньшей потерей информации)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2).fit(X_train_scaled) # оставляем 2 главные компоненты, 
                                                #чтобы данные можно было отрисовать график

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Форма исходного массива: {}".format(str(X_train_scaled.shape)))
print("Форма массива после сокращения размерности: {}".format(str(X_train_pca.shape)))


# In[435]:


# строим график 
plt.figure(figsize=(8, 8))
plt.scatter(X_train_pca[:,0],X_train_pca[:,1], c = y_train, alpha= 0.5)

plt.legend(cancer.target_names)

plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")


# In[436]:


# Теперь мы можем использовать новое представление, чтобы классифицировать данные
# используя классификатор LinearSVC

from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0)
clf.fit(X_train_pca,y_train)

print("Правильность на обучающем наборе: {:.2f}".format(clf.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test_pca, y_test)))


# In[437]:


# метод k ближайших соседей

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train_pca,y_train)
print("Правильность на обучающем наборе: {:.2f}".format(knn.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test_pca, y_test)))


# In[438]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for model, ax in zip([LinearSVC(), LogisticRegression(C = 4.6),KNeighborsClassifier(n_neighbors=1)], axes):
    clf = model.fit(X_train_pca, y_train)
    mglearn.plots.plot_2d_separator(clf, X_train_pca, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Признак 0")
    ax.set_ylabel("Признак 1")
axes[0].legend()
plt.show()


# In[444]:


# Отрисуем метод одного ближайшего соседа более подробно

from sklearn import neighbors
n_neighbors = 1

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max()) 
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification with using features from PCA(k = %i)"
% (n_neighbors))

plt.show()


# In[ ]:


# При методе  k ближайших соседей  модель хорошо предсказывает данные на тренировочных данных (100%), 
# но имеет более низкую обобщающую способность (90%). По графикам видно, что модель переобучена.

# На графиках также видно, что метод опорных векторов и логистическая регрессия не сильно отличаются и должны довать схожий результат.
# Так оно и есть. 
# При методе опорных векторов на тестовой выборке правильность = 96%, на тренировочной = 93%.
# При линейной регрессии на тестовой выборке правильность = 97%,  на тренировочной= 94%.
# Таким образом, использование логистической регрессии для классификации данного датасета яв-ся более целесообразным.


# ### leave-one-out

# In[445]:


X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, 
                                                         stratify = cancer.target, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval,y_trainval, random_state = 0)



from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
log_reg = LogisticRegression(C = 4.6)
scores = cross_val_score(log_reg, X_trainval,y_trainval, cv=loo)
print("Количество итераций: ", len(scores))
print("Средняя правильность: {:.2f}".format(scores.mean()))


# # GaussianNB

# In[446]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify = cancer.target, random_state = 0)


# In[447]:


# стандартиируем данные
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[448]:


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train_scaled, y_train)

print("Правильность на обучающем наборе: {:.2f}".format(clf.score(X_train_scaled, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test_scaled, y_test)))

