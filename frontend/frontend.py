import streamlit as st
import pandas as pd
import numpy as np
import backend
import datetime
import re
import matplotlib.pyplot as plt
import altair as alt
from etna.datasets import TSDataset
from etna.analysis import plot_trend
from etna.transforms import LinearTrendTransform
from etna.analysis import (
    cross_corr_plot,
    distribution_plot,
    acf_plot,
    plot_correlation_matrix,
    plot_feature_relevance,
    plot_backtest
)
import sys
import pandas as pd
from io import StringIO
import ast
import pathlib
from etna.core import load




st.header('Прогнозирование')

# Создание фильтров

# This code block creates filters for the user to select specific data to be displayed on the graph.
# It creates a sidebar with several input options such as date, website, source, and segment. The
# `apply` button is used to apply the selected filters to the data.
st.sidebar.title('Фильтры')
dfilt = st.sidebar.date_input('Дата', value=pd.Timestamp('2022-01-01'))
ufilt = st.sidebar.text_input('Сайт', value='ba.hse.ru/$')
sfilt = st.sidebar.multiselect('Источник', ['direct', 'organic', 'referral', 'internal', 'social', 'email', 'recommend', 'ad', 'saved', 'messenger','qrcode', 'undefined'], ['direct'])
segfilt = st.sidebar.text_input('Сегмент', value='ba.hse.ru/$')
HORIZON = st.sidebar.slider("Горизонт", 14, 365)
apply = st.sidebar.button('Применить фильтр')

# Общий график

@st.cache_data
def plot(df):
    chart = alt.Chart(df).mark_line().encode(
        x = 'timestamp:T',
        y = 'users:Q',
        color = 'segment:N'
    ).properties(
        width=500,
        height=300,
        title= f'График посещений сайта на {dfilt}'
    ).interactive()
    return chart


@st.cache_data
def fetch_data(dfilt, ufilt, sfilt):
    data = backend.get_data(str(dfilt), ufilt, sfilt)
    return data

# Датафрейм из backend
# `fin = backend.get_data(str(dfilt), ufilt, sfilt)` is calling a function `get_data` from the
# `backend` module and passing it the `dfilt`, `ufilt`, and `sfilt` variables. This function retrieves
# data from a database or other source based on the selected filters and returns a pandas DataFrame
# `fin` containing the relevant data.
fin = fetch_data(str(dfilt), ufilt, sfilt)

# Построение общего графика
# This code block is creating a chart using the `altair` library based on the data in the `fin`
# DataFrame, which is filtered based on user-selected filters for date, website, source, and segment.
# The `plot` function takes the DataFrame as input and returns an `altair` chart object. The
# `st.altair_chart` function is then used to display the chart in the Streamlit app with the
# `use_container_width=True` argument to make the chart responsive to the width of the app window.
ts, frog1 = backend.tsdataset_new(fin, segfilt)
frog1 = frog1.rename(columns={'target': 'users'})
chart = plot(frog1)
st.altair_chart(chart, use_container_width=True)

# Преобразование fin в формат ts

# These lines of code are creating checkbox inputs in the Streamlit app with labels "Построить график
# трендов", "Построить график сезонности", "Построить график аномалий", "Построить график прогноза",
# and "Построить пайплайн". If the user checks any of these checkboxes, the corresponding code block
# under it will be executed to generate the relevant chart or perform the relevant analysis. The
# resulting chart or output will be displayed in the Streamlit app.

with st.sidebar.expander('Список графиков', expanded=True):
    show_trend = st.checkbox('Построить график трендов')
    show_seasonality = st.checkbox('Построить график сезонности')
    show_anomalies = st.checkbox('Построить график аномалий')
    show_fit_model = st.checkbox('Обучить модель')
    show_backtest = st.checkbox('Построить backtest')
    show_forecast = st.checkbox('Построить график прогноза')
    show_hyperparams = st.checkbox('Подобрать гиперпараметры')


def plot_trend_chart(_ts, trend_degree):
    """
    The function plots a trend chart for a given time series data with a specified degree of trend.
    
    :param _ts: The time series data that we want to plot the trend chart for
    :param trend_degree: The degree of the polynomial used to fit the trend line to the time series
    data. A higher degree will result in a more flexible trend line that can capture more complex
    patterns in the data, but may also be more prone to overfitting
    :return: a trend chart generated using the input time series (_ts) and the degree of the trend
    (trend_degree).
    """
    trend_chart = backend.ts_trend(_ts, trend_degree)
    return trend_chart


# The above code is creating a slider widget to select the degree of a polynomial trend line to be
# plotted on a time series chart. If the `show_trend` variable is True, an expander widget is
# displayed with a header "График тренда" (which translates to "Trend Chart" in English). The user can
# select the degree of the polynomial trend line using the slider widget. The `plot_trend_chart`
# function is called to generate the trend chart based on the selected degree, and the chart is
# displayed using the `st.pyplot` function.
if show_trend:
    with st.expander('Тренд', expanded=True):
        st.header('График тренда')
        trend_degree = st.slider('Полином', min_value= 1, max_value = 4, value = 2)
        trend_chart = plot_trend_chart(ts, trend_degree)
        st.pyplot(trend_chart)


def plot_seasonality_chart(_ts, cycle):
    """
    This function plots a seasonality chart for a given time series and cycle.
    
    :param _ts: It is a time series data that we want to analyze for seasonality. It could be a list or
    an array of numerical values representing the data points over time
    :param cycle: The cycle parameter represents the length of the seasonal pattern in the time series
    data. For example, if the time series data exhibits a seasonal pattern that repeats every 12 months,
    then the cycle parameter would be set to 12
    :return: a seasonality chart for a given time series and cycle.
    """
    seasonality_chart = backend.get_seasonality(_ts, cycle)
    return seasonality_chart

# The above code is creating a section in a Streamlit app that allows the user to view a chart of the
# seasonality of a time series data. The user can choose the cycle (month, quarter, or year) for the
# seasonality chart to be plotted. The `plot_seasonality_chart` function is used to generate the chart
# and it is displayed using `st.pyplot`.
if show_seasonality:
    with st.expander('Сезонность', expanded=True):
        st.header('График сезонности')

        #Построение графика сезонности
        cycle_choice = ['month', 'quarter', 'year']
        cycle = st.radio('Цикл', cycle_choice)

        seasonality_chart = plot_seasonality_chart(ts, cycle)
        st.pyplot(seasonality_chart)


def plot_get_anomalies_chart(_ts, window, neighbours, distance):
    """
    This function takes a time series, window size, number of neighbors, and distance metric as inputs,
    and returns a list of anomalies detected by a backend algorithm.
    
    :param _ts: The time series data that we want to analyze for anomalies
    :param window: The window parameter is used to define the size of the sliding window used to
    calculate the moving average of the time series data. This moving average is then used to detect
    anomalies in the time series
    :param neighbours: The "neighbours" parameter likely refers to the number of neighboring data points
    that are considered when detecting anomalies. This can help to identify anomalies that are
    significantly different from the surrounding data points
    :param distance: Distance is a parameter used in anomaly detection algorithms to measure the
    similarity or dissimilarity between data points. It is often used in clustering algorithms to group
    similar data points together. In the context of this function, distance is likely used to determine
    which data points are considered anomalies based on their distance from the
    :return: the anomalies detected in a time series data using the parameters window, neighbours, and
    distance.
    """
    anomalies = backend.get_anomalies(_ts, window, neighbours, distance)
    return anomalies


def plot_fix_anomalies_chart(_ts, window, salpha, sstrategy, swindow_outlier):
    """
    This function takes in a time series, applies anomaly detection and correction techniques, and
    returns the corrected time series and the indices of the detected anomalies.
    
    :param _ts: This is a time series data that needs to be processed for fixing anomalies
    :param window: The size of the sliding window used to detect anomalies. It is an integer value
    :param salpha: salpha is a parameter that controls the significance level for detecting anomalies.
    It is a value between 0 and 1, where a smaller value indicates a higher sensitivity to anomalies
    :param sstrategy: sstrategy is a parameter that specifies the strategy to use for fixing anomalies
    in the time series data. It could be one of the following strategies:
    :param swindow_outlier: swindow_outlier is a parameter that specifies the size of the sliding window
    used to detect outliers in the time series data. It is used in the backend.fix_anomalies function to
    identify and fix anomalies in the time series
    :return: two values: `ts_fixed` and `pi`.
    """
    ts_fixed, pi = backend.fix_anomalies(_ts, window, salpha, sstrategy, swindow_outlier)
    return ts_fixed, pi
# This code block creates a checkbox input with label "Построить график аномалий". If the user checks
# this checkbox, the code block under it will be executed. It creates a header with text "Поиск
# аномалий", and several input widgets such as number inputs, selectbox, and a button. These widgets
# allow the user to specify parameters for anomaly detection such as window size, number of neighbors,
# distance coefficient, alpha, strategy, and window size for outlier replacement. When the user clicks
# the "Сгенерировать график аномалий" button, the `backend.get_anomalies` function is called with the
# specified parameters to detect anomalies in the time series data. The resulting chart is displayed
# using `st.pyplot`. The `backend.fix_anomalies` function is then called with the detected anomalies
# and the specified parameters for fixing the anomalies. The resulting chart is also displayed using
# `st.pyplot`.
if show_anomalies:
    with st.expander('График аномалий', expanded=True):
        st.header('Поиск аномалий')
        col1, col2 = st.columns(2)
        with col1:
            window = st.number_input('Размер окна', value=100, min_value=1, help = 'Размер скользящего окна, который определяет количество точек данных, которые рассматриваются одновременно для обнаружения аномалий')
        with col2:
            neighbours = st.number_input('Количество соседей', value=5, min_value=1, help = 'Количество соседних точек, которые необходимо учитывать при расчете плотности каждой точки во временном ряду')
        col3, alpha = st.columns(2)
        with col3:
            distance = st.number_input('Коэфициент дистанции', value=0.1, min_value=0.01, help = 'Параметр того, насколько далеко точка данных должна находиться от остальных данных, чтобы считаться аномалией')
        with alpha:
            salpha = st.number_input('Альфа', value=1.5, min_value=0.1, help = 'Параметр, используемый для определения порога обнаружения выбросов. Это значение от 0 до 1, где более высокое значение означает, что большее количество точек данных будет считаться выбросами')
        strategy, window_outlier = st.columns(2)
        with strategy:
            sstrategy = st.selectbox('Стратегия', ('mean', 'running_mean', 'forward_fill', 'seasonal'), help = 'Стратегии вменения, используемой для заполнения пропущенных значений в данных временного ряда')
            if sstrategy == 'mean':
                st.write('Пропущенные значения заменяются  средним значением ряда')
            elif sstrategy == 'running_mean':
                st.write('Пропущенные значения заменяются скользящим средним')
            elif sstrategy == 'forward_fill':
                st.write('Пропущенные значения заменяются последним известным значением во временном ряду')
            elif sstrategy == 'seasonal':
                st.write('Пропущенные значения заменяются значениями того же временного сезона в предыдущих периодах')
        with window_outlier:    
            swindow_outlier = st.number_input('Размер окна для замены выбросов', value=5, min_value=1, help = 'Размер окна, используемый для вменения пропущенных значений во временном ряду с использованием указанной стратегии')
        anomalies = plot_get_anomalies_chart(ts, window, neighbours, distance)
        st.pyplot(anomalies)
        ts_fixed, pi = plot_fix_anomalies_chart(ts, window, salpha, sstrategy, swindow_outlier)
        st.pyplot(pi)


pipeline = None
pipeline_loaded = None
model_trained = False
list_transforms = []
SAVE_DIR = pathlib.Path("tmp")
SAVE_DIR.mkdir(exist_ok=True)
def train_pipeline(ts, HORIZON, list_transforms):
    pipeline, _ = backend.pipeline_train(ts, HORIZON, list_transforms)
    return pipeline

#@st.cache_data()
def load_pipeline(ts, pipeline):
    pipeline.save(SAVE_DIR / "pipeline.zip")
    pipeline_loaded = load(SAVE_DIR / "pipeline.zip", ts=ts)
    return pipeline_loaded


if show_fit_model:
    def fit():            
        st.header('Обучение модели')
        with st.expander('Обучение модели', expanded=True):
            transforms= ['scale', 'yeoj_transf', 'date_flags', 'scaler', 'fourier', 'holidays', 'holidays_lags', 'imputer', 'lin_trend', 'trend_transf', 'mean_tr', 'lags_month', 'lags_week', 'log']
            transforms_without_quotes =( '[%s]' % ', '.join(map(str, transforms)))
            list_transforms = st.multiselect('Выбор трансформа', transforms, ['scale', 'yeoj_transf', 'date_flags'])
        
            #retrain_model = st.button('Переобучить модель')
            #if retrain_model:
            pipeline = train_pipeline(ts, HORIZON, list_transforms)

            if pipeline:
                st.success('Модель обучена!')
                model_trained = True
                pipeline_loaded = load_pipeline(ts, pipeline)
            else:
                st.info('Ошибка выполнения или выберите трансформ')
                model_trained = False

            #else:
            #    st.warning('Обучите модель')

        #pipeline.save(SAVE_DIR / "pipeline.zip")
        #list_transforms[0].save(SAVE_DIR / "transform_0.zip")
        #pipeline_loaded = load(SAVE_DIR / "pipeline.zip", ts=ts)
        
        return pipeline_loaded, model_trained, list_transforms
        #transform_0_loaded = load(SAVE_DIR / "transform_0.zip")
    #fit_result = fit()
    pipeline_loaded, model_trained, list_transforms = fit()



# The above code is a Python code block that checks if the `show_backtest` flag is set to `True` and
# if the `model_trained` flag is also `True`. If both conditions are met, it displays a header for the
# backtest section and creates an expander widget for the backtest section. Within the expander
# widget, it allows the user to input the number of folds for the backtest and then calls the
# `backend.backtest()` function to perform the backtest on the time series data using the specified
# pipeline and number of folds. It then displays the feature relevance
if show_backtest:
    if model_trained:
        st.header('Backtest')
        with st.expander('Backtest', expanded=True):
            N_FOLDS = st.number_input('Количество сгибов', value = 1, min_value = 1, max_value = 1700)
            model_relevance, model_relevance_table, metrics_df_head, forecast_df = backend.backtest(ts, pipeline_loaded, N_FOLDS)
            
            plot_feature_relevance(
                ts=ts,
                relevance_table=model_relevance_table,
                relevance_aggregation_mode="mean",
                relevance_params={"model": model_relevance},
                top_k=20
                        )
            st.pyplot()
            #st.dataframe(model_relevance_table)
            plot_backtest(forecast_df, ts=ts, history_len=HORIZON)
            st.pyplot()
            #model_relevance      
            st.write('Метрики', metrics_df_head)
    else:
        st.warning('Сначала обучите модель')



def plot_forecast_chart(pipeline, n_train_samples, ts, HORIZON):
    """
    This function generates a forecast chart using a given pipeline, number of training samples, time
    series data, and horizon.
    
    :param pipeline: It is a machine learning pipeline that has been trained on historical time series
    data to make forecasts
    :param n_train_samples: The number of samples used for training the forecasting model
    :param ts: The time series data used for making the forecast
    :param HORIZON: The number of time steps to forecast into the future
    :return: a tuple containing two values: the forecast plot and the upper bounds of the forecast.
    """
    forecast_plot, yu, _ = backend.make_forecast(pipeline, n_train_samples, ts, HORIZON)
    return forecast_plot, yu 

# The above code is a Python code block that checks if the `show_forecast` variable is True and if the
# `model_trained` variable is also True. If both conditions are met, it displays an expander widget
# with a chart and a number input field. The chart is generated by calling the `plot_forecast_chart`
# function with some arguments, and the resulting chart is displayed using the `st.pyplot` function.
# The `st.write` function is also used to display some text. If the `model_trained` variable is False,
# a warning message is displayed using the `st
if show_forecast:
    if model_trained:
        st.header('Прогноз')
        with st.expander('Прогноз', expanded=True):
            n_train_samples = st.number_input('Количество выборок, используемых для обучения модели', value = 40, min_value = 1, max_value = 1700)
            forecast_plot, yu = plot_forecast_chart(pipeline_loaded, n_train_samples, ts, HORIZON)
            st.pyplot(forecast_plot)
            st.write(yu)
    else:
        st.warning('Сначала обучите модель')

if show_hyperparams:
    with st.expander('Гиперпараметры', expanded=True):
        list_transforms
        model = backend.tune_parametres(HORIZON, list_transforms, ts)
        hypplot, prediction = backend.new_prediction(model, n_train_samples, ts, HORIZON)
        st.pyplot(hypplot)
        st.write(prediction)






