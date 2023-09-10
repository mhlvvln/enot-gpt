
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Чтение данных
df = pd.read_csv('clear_dataset_k.csv', low_memory=False)

df_positive = df[df['target'] == 1]
df_negative = df[df['target'] == 0]

sample_size_positive = 436
sample_size_negative = 561

df_positive_sampled = df_positive.sample(n=sample_size_positive)
df_negative_sampled = df_negative.sample(n=sample_size_negative)

balanced_df = pd.concat([df_positive_sampled, df_negative_sampled])

# Определение категориальных признаков
categorical_features = balanced_df.select_dtypes(include=['object']).columns.tolist()
categorical_features = [feat for feat in categorical_features if feat != 'target']

X = balanced_df.drop(['target'], axis=1)
y = balanced_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# pool для тестиовой выбокри
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

class_weights = [1, 2]
model = CatBoostClassifier(iterations=6, depth=11, learning_rate=0.8, loss_function='Logloss', class_weights=class_weights, verbose=True)
model.fit(train_pool, eval_set=test_pool)

# оценим проиводительность
y_pred = model.predict(test_pool)
print(classification_report(y_test, y_pred))

# Сохранение модели
model.save_model('model_name')