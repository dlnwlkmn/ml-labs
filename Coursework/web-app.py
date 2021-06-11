import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def get_metricts_for_alg(self, metrics, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        to_return = list()
        for metric in metrics:
            temp_data = self.df[self.df['metric']==metric]
            temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
            to_return.append((temp_data_2['alg'].values, metric, temp_data_2['value'].values))
        return to_return


    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        figsize = (5, 1*len(array_labels))
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,4)), color='white')
        st.pyplot(fig)  

    def abs_plot(self, metrics, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_metric = list()
        metrics_data_list = self.get_metricts_for_alg(metrics, ascending)
        for i in metrics_data_list:
            array_metric.append(i[2])
        st.text(metrics_data_list)
        figsize = (5, 1*len(metrics_data_list))
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric[0]))
        rects = ax1.barh(pos, metrics_data_list[0][0],
                         align='center',
                         height=0.5, 
                         tick_label=metrics)
        ax1.set_title(metrics_data_list[0][0])
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white')
        st.pyplot(fig)  

# ЗАГРУЗКА ДАННЫХ
@st.cache
def load_data():
    data_wine = pd.read_csv('/Users/savelevaa/Desktop/ml-labs/Coursework/data/wine.csv', sep=",")
    return data_wine

# функция классификации целевого пизнака
def type_to_int(w_type: str) -> int:
    if w_type == "white":
        result = 0
    elif w_type == "red":
        result = 1
    return result 

def scaler(data):
    # Получим названия колонок
    data_cols = list()
    temp_cols = data.columns
    for col in temp_cols:
        data_cols.append(col)
    data_cols.pop(0)
    sc = MinMaxScaler()
    scaled_data = sc.fit_transform(data[data_cols])
    # Добавим масштабированные данные в набор данных
    for i in range(len(data_cols)):
        col = data_cols[i]
        new_col_name = col + ' scaled'
        data[new_col_name] = scaled_data[:,i]
    return data
    

@st.cache
def prepare_data(data):
    # Оставим только непустые значения
    for col in data.columns:
        data = data[data[col].notna()]
    data['target'] = \
        data.apply(lambda row: type_to_int(row['type']), axis=1)
    data = scaler(data)
    # Признаки для задачи классификации
    task_clas_cols = ['volatile acidity scaled', 'density scaled', 
                  'sulphates scaled', 'chlorides scaled', 'fixed acidity scaled']
    # Выборки для задачи классификации
    cl_X_train, cl_X_test, cl_Y_train, cl_Y_test = train_test_split(
        data[task_clas_cols], data['target'].values, test_size=0.5, random_state=1)
    return (cl_X_train, cl_X_test, cl_Y_train, cl_Y_test)


# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, 
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='cyan',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

@st.cache(suppress_st_warning=True)
def clas_train_model(model_names, X_train, X_test, Y_train, Y_test, clasMetricLogger=None, Kn=None):
    current_models_list = []
    roc_auc_list = []

    for model_name in model_names:
        model = None
        if Kn == None:
            model = clas_models[model_name]
        else:
            model = KNeighborsClassifier(n_neighbors=Kn)
        model.fit(X_train, Y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:,1]

        roc_auc = roc_auc_score(Y_test, Y_pred_proba)

        if clasMetricLogger != None:
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred)
            recall = recall_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred)
            clasMetricLogger.add('accuracy', model_name, accuracy)
            clasMetricLogger.add('precision', model_name, precision)
            clasMetricLogger.add('recall', model_name, recall)
            clasMetricLogger.add('f1', model_name, f1)
            clasMetricLogger.add('roc_auc', model_name, roc_auc)

        roc_auc = roc_auc_score(Y_test, Y_pred_proba)
        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        #Отрисовка ROC-кривых 
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
        draw_roc_curve(Y_test, Y_pred_proba, ax[1])
        plot_confusion_matrix(model, X_test, Y_test, ax=ax[0],
                        display_labels=['0','1'], 
                        cmap=plt.cm.Oranges, normalize='true')
        fig.suptitle(model_name)
        st.pyplot(fig)

        # if Kn != None:
        #     for metric in list(clasMetricLogger.df['metric'].unique()):
        #         clasMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6))
        #     # clasMetricLogger.abs_plot(list(clasMetricLogger.df['metric'].unique()), figsize=(7, 6))

st.sidebar.header('Управление:')


info = st.sidebar.checkbox('Информация по набору данных')
corr = st.sidebar.checkbox('Показать корреляционную матрицу')
print_models = st.sidebar.checkbox('Модели машинного обучения')
gparams = st.sidebar.checkbox('Подбор гиперпараметров')
absolute = st.sidebar.checkbox('Модель K-ближайших соседей')
evaluation = st.sidebar.checkbox('Оценки качества моделей')

# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}

params_list = ['GridSearch', 'RandomSearch']

# Создаем хранитель метрик
metricLogger1 = MetricLogger()

#st.sidebar.header('Параметры:')

st.header('Курсовой проект по дисциплине "Технологии машинного обучения"')
st.subheader('Веб-приложение с использованием фреймворка streamlit')
st.subheader('Статус:')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

train_test_tpl = prepare_data(data)


if info:
    st.text('Учебный набор данных библиотеки sklearn digits для решения задачи классификации')
    st.text(f'Размерность: строки: {data.shape[0]}, колонки: {data.shape[1]}')
    st.subheader('Данные:')
    st.write(data)

if corr:
    fig1, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(data.corr(), fmt='.2f', annot=True)
    st.pyplot(fig1)
 
if print_models:
    st.sidebar.header('Модели машинного обучения')
    st.subheader('Модели:')
    models_select = st.sidebar.multiselect('Выберите модели', models_list)
    clas_train_model(models_select, train_test_tpl[0], train_test_tpl[1],
                    train_test_tpl[2], train_test_tpl[3], metricLogger1)

if gparams:
    st.subheader('Подбор гиперпараметров:')
    st.sidebar.header('Параметры:')
    param_select = st.sidebar.multiselect('Выберите метод подбора:', params_list)
    step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=100, value=20, step=1)
    max_border_neighbor = st.sidebar.slider('Макс кол-во соседей:', min_value=50, max_value=1000, value=200, step=5)

    #Количество записей
    data_len = data.shape[0]
    st.write('Количество строк в наборе данных - {}'.format(data_len))
    # Подбор гиперпараметра
    n_range_list = list(range(3,max_border_neighbor, step_slider))
    n_range = np.array(n_range_list)
    st.write('Возможные значения соседей - {}'.format(n_range))
    tuned_parameters = [{'n_neighbors': n_range}]

    

    st.subheader('Оценка качества модели')
    if "GridSearch" in param_select:
        clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_gs.fit(train_test_tpl[0], train_test_tpl[2])

        st.subheader("Grid Search")
        st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))
        # Изменение качества на тестовой выборке в зависимости от К-соседей
        fig1, ax1 = plt.subplots(figsize=(10,5)) 
        ax1 = plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
        st.pyplot(fig1)
        
    if "RandomSearch" in param_select:
        clf_rs = RandomizedSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_rs.fit(train_test_tpl[0], train_test_tpl[2])

        st.subheader("Random Search")
        st.write('Лучшее значение параметров - {}'.format(clf_rs.best_params_))
        # Изменение качества на тестовой выборке в зависимости от К-соседей
        fig1, ax1 = plt.subplots(figsize=(10,5))    
        ax1 = plt.plot(n_range, clf_rs.cv_results_['mean_test_score'])
        st.pyplot(fig1)

metricLogger2 = MetricLogger()

if absolute:
    st.sidebar.header('Модель К-ближайших соседей:')
    st.subheader("Модель К-ближайших соседей")
    neighbors_num = st.sidebar.slider('K-ближайших соседей:', min_value=1, max_value=100, value=20, step=1)
    clas_train_model([f"KNN_{neighbors_num}"], train_test_tpl[0], train_test_tpl[1],
                    train_test_tpl[2], train_test_tpl[3], metricLogger1, neighbors_num)


if evaluation:
    # Метрики качества модели
    clas_metrics = list(metricLogger1.df['metric'].unique())

    if len(clas_metrics) == 0:
        st.subheader('Оценок качества моделей нет')
    else:
        st.subheader('Оценки качества моделей:')
        # Построим графики метрик качества модели
        for metric in clas_metrics:
            metricLogger1.plot('Метрика: ' + metric, metric, figsize=(7, 6))