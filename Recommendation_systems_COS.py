import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances

# Загружаем датасет
df = pd.read_csv('ratings.csv')  # 'ratings_mini.csv' 'ratings.csv'

# Выводим количество пользователей и фильмов
n_users = df['userId'].unique().shape[0]
n_items = df['productId'].unique().shape[0]
print('Число пользователей', n_users)
print('Число продуктов', n_items)
print(df)

# Plot the ratings to view distribution of the same
fig, ax = plt.subplots(1, 1, figsize=(4, 6))
sns.distplot(df.rating, color="blue", ax=ax, kde=False)
ax.set_title("Histogram of Ratings", fontsize=20)
plt.tight_layout()
plt.show()

# Создаём user - product матрицу, отсутствующие рейтинги заменяем нулями
data_matrix = np.zeros((n_users, n_items))
for line in df.itertuples():
    data_matrix[line[1] - 1, line[2] - 1] = line[3]
print('data_matrix:')
print(data_matrix)

# Предсказываем рейтинги - заменяем ими нули - в матрице данных
# ========== алгоритм - косинусное расстояние для пользователей и товаров
# Считаем косинусное расстояние для пользователей и товаров
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
user_similarity = np.round(user_similarity, 1)
item_similarity = np.round(item_similarity, 1)
print('user_similarity', user_similarity)
print('item_similarity', item_similarity)


def predict(ratings, similarity, type='user'):
    global pred
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = (mean_user_rating[:, np.newaxis] +
                + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T)
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


item_prediction = predict(data_matrix, item_similarity, type='item')
user_prediction = predict(data_matrix, user_similarity, type='user')  # Заменяем нули рейтингами
user_prediction = np.round(user_prediction, 1)
item_prediction = np.round(item_prediction, 1)
print('user_prediction')
print(user_prediction)
print('item_prediction')
print(item_prediction)
print('User-based CF RMSE: ' + str(rmse(user_prediction, data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, data_matrix)))
predict_matrix = user_prediction

# ==========закончили расчет рейтингов, используя косинусные расстояния=============
predict_matrix_T = predict_matrix.T  # Транспонируем

# Вывод матрицы с прогнозами в красивом виде (транспониуем)
fig, ax = plt.subplots()
M = predict_matrix_T[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
ax.matshow(M, cmap=plt.cm.Reds)
for i in range(16):
    for j in range(10):
        c = predict_matrix_T[j, i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.xlabel('Пользователи')
plt.ylabel('Товары')
plt.title('Фрагмент предсказанных рейтингов от Жилинского К. В. COS')
plt.show()

# Считаем средние рейтинги для каждого товара
ind_avg = np.mean(predict_matrix_T[:, :n_items], axis=1)
print('Средние рейтинги:', ind_avg)

# Сортировка по убыванию всех средних. Сохраняем только номер
n = len(ind_avg)
ind_sort = [0] * n
R_sort = [0] * n
ind = list(range(n))
for i in range(n):
    for j in range(i+1, n):
        if ind_avg[ind[i]] < ind_avg[ind[j]]:
            ind[i], ind[j] = ind[j], ind[i]
print(ind_avg)
for i in range(n):
    ind_sort[i] = ind[i]+1
    R_sort[i] = np.round(ind_avg[ind[i]], 1)
print('Номера продуктов, упорядоченных по рейтингу', ind_sort)
print('Упорядоченные средние рейтинги', R_sort)

# Вывод 5 наиболее рейтинговых - можно другое число
ind_sort_max = ind_sort[:5]
R_sort_max = R_sort[:5]
print('Наиболее рейтинговые', ind_sort_max)
print('Их средние рейтинги', R_sort_max)

# Вывод названий
# Load in product information data
products = pd.read_csv('products.csv')
print(products)
print('Продукты, которые вызвали интерес у покупателя (самый большой рейтинг):', ind_sort_max)

# Create array containing the names of the top products
top_products_with_name = []
for prod in ind_sort_max:
    df_filt = products.loc[(products.product_id == int(prod))]
    top_products_with_name.append(df_filt.title.values[0])
top_products_with_name = np.array(top_products_with_name)
print("Названия этих продуктов:", top_products_with_name)
