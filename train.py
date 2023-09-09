import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report


df = pd.read_csv('clear_dataset_k.csv', low_memory=False)

df_positive = df[df['target'] == 1]
df_negative = df[df['target'] == 0]

sample_size_positive = 436
sample_size_negative = 436*3

df_positive_sampled = df_positive.sample(n=sample_size_positive)
df_negative_sampled = df_negative.sample(n=sample_size_negative)

balanced_df = pd.concat([df_positive_sampled, df_negative_sampled])

categorical_features = balanced_df.select_dtypes(include=['object']).columns.tolist()
categorical_features = [feat for feat in categorical_features if feat != 'target']

X = balanced_df.drop(['target'], axis=1)
y = balanced_df['target']

train_pool = Pool(X, y, cat_features=categorical_features)

class_weights = [1, 2]  # Устанавливаем веса классов
model = CatBoostClassifier(iterations=13, depth=10, learning_rate=0.1, loss_function='Logloss', class_weights=class_weights, verbose=True)
model.fit(train_pool)

y_pred = model.predict(X)
print(classification_report(y, y_pred))

model.save_model('model_name')
