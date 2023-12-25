import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def some_predicts(data):
    # Удалите целевую переменную (target)
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']

    # Добавим виджет для настройки параметра test_size
    test_size = st.slider("Выберите размер тестового набора:", 0.1, 0.5, 0.25, 0.01)

    # Разделите данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Инициализируйте несколько классификаторов
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'Logistic Regression': LogisticRegression()
    }

    # Обучите и оцените производительность каждого классификатора
    results = {}
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, zero_division=1)
        results[name] = {'accuracy': accuracy, 'classification_report': classification_rep}

    # Выведите результаты
    for name, result in results.items():
        st.subheader(name)
        st.write(f"Accuracy: {result['accuracy']:.2f}")
        st.text(f"Classification Report\nchr {result['classification_report']}")

def custom_eda(data):
    # Реализуйте здесь ваш кастомный EDA
    # Например, вы можете использовать методы pandas для анализа данных

    # В данном случае просто выводим основные статистики
    st.write("Основные статистики данных:")
    st.write(data.describe())

    # Выведите распределения всех признаков в виде каскада графиков
    for column in data.columns:
        st.subheader(f'Распределение {column}')
        st.pyplot(sns.histplot(data[column], kde=True, color='skyblue').figure)

    corr = data.corr()
    st.subheader('Матрица корреляции')
    st.pyplot(sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5).figure)

    # Выведите виджеты для выбора признаков
    selected_feature = st.selectbox("Выберите признак, чтобы построить график распределения:", data.columns)
    target_variable = 'TARGET'

    # Определите количество уникальных значений в выбранном признаке
    unique_values_count = data[selected_feature].nunique()

    # Выберите тип графика в зависимости от количества уникальных значений
    if unique_values_count <= 10:
        # Гистограмма для малого количества уникальных значений
        st.subheader(f'Распределение {selected_feature} по классам {target_variable}')
        st.pyplot(sns.countplot(x=selected_feature, hue=target_variable, data=data, palette='pastel', dodge=True).figure)
    else:
        # Обычный график для большого количества уникальных значений
        st.subheader(f'Boxplot {selected_feature} по классам {target_variable}')
        st.pyplot(sns.boxplot(x=target_variable, y=selected_feature, data=data, hue=target_variable, palette='pastel', legend=False).figure)

    some_predicts(data)

def main():
    st.title("Кастомный EDA с Streamlit")

    uploaded_file = st.file_uploader("Загрузите свой датасет (иначе будет использован базовый) (CSV)", type="csv")

    # Если пользователь загрузил файл, прочитайте его в датафрейм
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # В противном случае используйте свой датасет (замените "ваш_файл.csv" на фактический путь)
        df = pd.read_csv('data.csv')

    # Отображение данных
    st.write("Первые 5 строк датафрейма:")
    st.write(df.head())

    # Выполнение кастомного EDA
    custom_eda(df)

if __name__ == "__main__":
    main()
