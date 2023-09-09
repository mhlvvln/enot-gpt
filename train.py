import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report

# Чтение данных
df = pd.read_csv('clear_dataset_k.csv', low_memory=False)

# Создание сбалансированного набора данных
df_positive = df[df['target'] == 1]
df_negative = df[df['target'] == 0]

# Указываем желаемые размеры обучающей выборки
sample_size_positive = 436
sample_size_negative = 436*3

df_positive_sampled = df_positive.sample(n=sample_size_positive)
df_negative_sampled = df_negative.sample(n=sample_size_negative)

balanced_df = pd.concat([df_positive_sampled, df_negative_sampled])

# Определение категориальных признаков
categorical_features = balanced_df.select_dtypes(include=['object']).columns.tolist()
categorical_features = [feat for feat in categorical_features if feat != 'target']

X = balanced_df.drop(['target'], axis=1)
y = balanced_df['target']

# Создание Pool
train_pool = Pool(X, y, cat_features=categorical_features)

# Инициализация и обучение модели
class_weights = [1, 2]  # Устанавливаем веса классов
model = CatBoostClassifier(iterations=13, depth=10, learning_rate=0.1, loss_function='Logloss', class_weights=class_weights, verbose=True)
model.fit(train_pool)

# Оценка производительности модели
y_pred = model.predict(X)
print(classification_report(y, y_pred))

# Сохранение модели
model.save_model('model_name')
