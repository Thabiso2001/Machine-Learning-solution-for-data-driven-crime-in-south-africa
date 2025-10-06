

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional: statsmodels for forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS = True
except Exception:
    STATSMODELS = False

st.set_page_config(layout='wide', page_title='Crime Dashboard')

# ----------------- Helper functions -----------------

def load_data(uploaded_file):
    # Try to read csv, excel, or parquet
    if uploaded_file is None:
        return None
    fname = getattr(uploaded_file, 'name', '')
    try:
        if fname.endswith('.csv') or fname.endswith('.txt'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif fname.endswith('.xlsx') or fname.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif fname.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            # attempt csv
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None
    return df


def ensure_datetime(df, date_col='date'):
    if date_col not in df.columns:
        st.warning(f"Date column '{date_col}' not found. Trying to auto-detect columns with 'date' in name.")
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c
                break
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.error(f"Failed to convert column {date_col} to datetime. Please provide a datetime column.")
        raise
    return df, date_col


def basic_relevance_check(df):
    # Quick automatic check of relevance of uploaded files to a crime dataset
    checks = []
    expected_cols = ['date','category','location','latitude','longitude']
    present = [c for c in expected_cols if any(c in col.lower() for col in df.columns)]
    checks.append(("expected_columns_present", present))
    # number of rows
    checks.append(("n_rows", len(df)))
    # timezone or date coverage
    datecols = [c for c in df.columns if 'date' in c.lower()]
    if datecols:
        try:
            dc = pd.to_datetime(df[datecols[0]], errors='coerce')
            checks.append(("date_range", (str(dc.min()), str(dc.max()))))
        except Exception:
            pass
    return checks


def plot_eda(df, date_col, category_col, location_col, lat_col=None, lon_col=None):
    st.subheader('Exploratory Data Analysis')
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown('**Crimes over time**')
        by_time = df.groupby(pd.Grouper(key=date_col, freq='M')).size().reset_index(name='count')
        fig = px.line(by_time, x=date_col, y='count', title='Monthly crime counts')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('**Top categories**')
        cat_counts = df[category_col].value_counts().reset_index()
        cat_counts.columns = [category_col,'count']
        fig2 = px.bar(cat_counts.head(20), x=category_col, y='count', title='Top categories')
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('**Heatmap: location vs category (counts)**')
        # create pivot table for top N categories and locations
        top_cats = df[category_col].value_counts().head(10).index
        top_locs = df[location_col].value_counts().head(10).index
        pivot = pd.pivot_table(df[df[category_col].isin(top_cats) & df[location_col].isin(top_locs)],
                               index=location_col, columns=category_col, values=date_col, aggfunc='count', fill_value=0)
        fig3 = px.imshow(pivot, labels=dict(x='Category', y='Location', color='Count'), title='Location vs Category heatmap')
        st.plotly_chart(fig3, use_container_width=True)

    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        st.markdown('**Map of incidents (sampled)**')
        sample = df[[lat_col, lon_col, category_col, date_col]].dropna().sample(min(2000, len(df)))
        fig4 = px.scatter_mapbox(sample, lat=lat_col, lon=lon_col, hover_data=[category_col,date_col], zoom=10)
        fig4.update_layout(mapbox_style='open-street-map')
        fig4.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
        st.plotly_chart(fig4, use_container_width=True)


def run_classification(df, category_col, feature_cols, test_size=0.2, random_state=42):
    st.subheader('Classification')
    # simple label encode target
    df = df.dropna(subset=[category_col])
    y = df[category_col].astype(str)
    X = df[feature_cols].copy()
    # simple preprocessing: fill numeric na with median, categorical with mode
    for c in X.columns:
        if X[c].dtype.kind in 'biufc':
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna('missing').astype(str)
    # one-hot encode small cardinality categoricals
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    report = classification_report(y_test, y_pred, output_dict=True)

    # display
    st.markdown('**Model:** RandomForestClassifier (200 trees)')
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')

    # If binary, show ROC AUC
    if len(model.classes_) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(pd.factorize(y_test)[0], y_proba)
            st.write(f'ROC AUC: {auc:.3f}')
        except Exception:
            pass

    st.markdown('**Confusion Matrix**')
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.markdown('**Classification report**')
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
    st.markdown('**Top feature importances**')
    st.bar_chart(importances)

    return model, X.columns


def forecast_time_series(series, periods=12):
    # series: pandas Series indexed by datetime
    series = series.asfreq('D').fillna(0)
    train = series
    if not STATSMODELS:
        # fallback: simple rolling mean forecast
        mean = train.mean()
        forecast = pd.Series(mean, index=pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit='D'), periods=periods))
        resid_std = series.diff().std()
        upper = forecast + 1.96 * resid_std
        lower = forecast - 1.96 * resid_std
        return forecast, lower, upper
    # Use ExponentialSmoothing
    try:
        model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=7).fit(optimized=True)
        pred = model.forecast(periods)
        # confidence intervals via residual std
        resid = model.resid.dropna()
        sigma = resid.std()
        ci = 1.96 * sigma
        upper = pred + ci
        lower = pred - ci
        return pred, lower, upper
    except Exception:
        # fallback
        mean = train.mean()
        forecast = pd.Series(mean, index=pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit='D'), periods=periods))
        resid_std = series.diff().std()
        upper = forecast + 1.96 * resid_std
        lower = forecast - 1.96 * resid_std
        return forecast, lower, upper

# ----------------- App layout -----------------

st.title('Crime Analysis & Forecast Dashboard')
st.markdown('Upload a crime dataset (CSV/Excel/Parquet). Expected useful columns: date, category, location, latitude, longitude, plus any explanatory features for classification.')

uploaded = st.file_uploader('Upload dataset', type=['csv','xlsx','xls','parquet','txt'])

# also try to load an auto-discovered dataset path if available (e.g., /mnt/data/FINAL_EXAM2025.ipynb won't be the dataset)
AUTO_PATH = '/mnt/data/crime.csv'
use_auto = False
try:
    import os
    if os.path.exists(AUTO_PATH):
        use_auto = st.checkbox(f'Also try to load {AUTO_PATH} if no upload provided', value=False)
        if use_auto and uploaded is None:
            with open(AUTO_PATH, 'rb') as f:
                uploaded = io.BytesIO(f.read())
                uploaded.name = 'crime.csv'
except Exception:
    pass

if uploaded is None:
    st.info('Please upload a dataset to begin. A sample dataset option is available below.')
    if st.button('Load sample dataset (synthetic)'):
        # create synthetic sample dataset
        n = 5000
        rng = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='H')
        df_sample = pd.DataFrame({
            'date': np.random.choice(rng, n),
            'category': np.random.choice(['Theft','Assault','Burglary','Robbery','Vandalism'], n),
            'location': np.random.choice(['Central','Westside','Eastside','North','South'], n),
            'latitude': np.random.uniform(-33.95, -33.80, n),
            'longitude': np.random.uniform(18.40, 18.70, n),
            'feature1': np.random.randn(n),
            'feature2': np.random.randint(0,5,n)
        })
        df_sample.to_csv('sample_crime.csv', index=False)
        uploaded = open('sample_crime.csv','rb')

if uploaded is not None:
    df = load_data(uploaded)
    if df is None:
        st.stop()

    # show relevance check
    st.sidebar.header('Data relevance check')
    checks = basic_relevance_check(df)
    for k,v in checks:
        st.sidebar.write(f"**{k}**: {v}")

    # datetime conversion
    try:
        df, date_col = ensure_datetime(df)
    except Exception:
        st.stop()

    # user chooses columns
    cols = df.columns.tolist()
    st.sidebar.header('Columns mapping')
    category_col = st.sidebar.selectbox('Crime category column', options=[c for c in cols if df[c].dtype==object or 'cat' in c.lower()], index=0 if any('category' in c.lower() for c in cols) else 0)
    location_col = st.sidebar.selectbox('Location column', options=[c for c in cols if df[c].dtype==object or 'loc' in c.lower()], index=0 if any('location' in c.lower() for c in cols) else 1 if len(cols)>1 else 0)
    lat_col = st.sidebar.selectbox('Latitude column (optional)', options=[None]+cols, index=0)
    lon_col = st.sidebar.selectbox('Longitude column (optional)', options=[None]+cols, index=0)

    # Filters
    st.sidebar.header('Filters')
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_range = st.sidebar.date_input('Date range', value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
    selected_categories = st.sidebar.multiselect('Categories', options=sorted(df[category_col].dropna().unique()), default=sorted(df[category_col].dropna().unique()))
    selected_locations = st.sidebar.multiselect('Locations', options=sorted(df[location_col].dropna().unique()), default=sorted(df[location_col].dropna().unique())[:10])

    # apply filters
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt) & (df[category_col].isin(selected_categories)) & (df[location_col].isin(selected_locations))
    df_filt = df.loc[mask].copy()
    st.write(f'Dataset after filters: {len(df_filt)} rows')

    # EDA
    plot_eda(df_filt, date_col, category_col, location_col, lat_col if lat_col else None, lon_col if lon_col else None)

    # Classification section
    st.sidebar.header('Classification settings')
    with st.expander('Classification (quick setup)'):
        st.write('Choose columns to use as features. Non-numerical columns will be one-hot encoded.')
        feature_cols = st.multiselect('Feature columns', options=[c for c in cols if c not in [date_col, category_col, location_col]], default=[c for c in cols if c not in [date_col, category_col, location_col]][:5])
        run_clf = st.button('Run classification')
        if run_clf:
            if len(feature_cols) < 1:
                st.error('Select at least one feature column for classification')
            else:
                try:
                    model, feat_cols = run_classification(df_filt, category_col, feature_cols)
                except Exception as e:
                    st.error(f'Classification failed: {e}')

    # Forecasting section
    st.header('Time-series forecasting')
    st.markdown('Select a grouping to produce a daily time series and forecast future counts with confidence intervals.')
    group_by = st.selectbox('Group incidents by', options=['category','location','none'], index=0)
    if group_by == 'category':
        group_choice = st.selectbox('Choose category', options=sorted(df_filt[category_col].unique()))
        series = df_filt[df_filt[category_col]==group_choice].set_index(date_col).resample('D').size()
    elif group_by == 'location':
        group_choice = st.selectbox('Choose location', options=sorted(df_filt[location_col].unique()))
        series = df_filt[df_filt[location_col]==group_choice].set_index(date_col).resample('D').size()
    else:
        series = df_filt.set_index(date_col).resample('D').size()

    st.line_chart(series)
    periods = st.number_input('Forecast horizon (days)', min_value=7, max_value=365, value=30)
    if st.button('Run forecast'):
        with st.spinner('Fitting forecast model...'):
            pred, lower, upper = forecast_time_series(series, periods=periods)
            # show
            fc_index = pred.index
            df_fc = pd.DataFrame({'forecast': pred, 'lower': lower, 'upper': upper})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name='historical'))
            fig.add_trace(go.Scatter(x=fc_index, y=df_fc['forecast'], name='forecast'))
            fig.add_trace(go.Scatter(x=fc_index, y=df_fc['upper'], name='upper', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=fc_index, y=df_fc['lower'], name='lower', line=dict(dash='dash')))
            fig.update_layout(title='Forecast with confidence intervals')
            st.plotly_chart(fig, use_container_width=True)

    # Summaries for technical and non-technical users
    st.header('Summaries')
    with st.expander('Technical summary'):
        st.markdown('''
        **Technical summary includes:**
        - Dataset size and date range
        - Filters applied
        - EDA figures (time series, top categories, heatmap)
        - Classification model and metrics (confusion matrix, classification report)
        - Forecast model used (ExponentialSmoothing if available) and prediction intervals computed from residual std.
        ''')
        st.write({'n_rows': len(df_filt), 'date_range': (str(start_dt.date()), str(end_dt.date())), 'n_categories': int(df_filt[category_col].nunique())})

    with st.expander('Non-technical summary'):
        st.markdown('''
        **Non-technical summary (plain language):**
        - The dashboard shows how many crimes occurred over time and where they happen most.
        - Use the filters on the left to focus on certain crime types, places, and time periods.
        - A simple machine learning model attempts to predict crime category from the features you selected — results are shown as accuracy and a confusion matrix.
        - The forecasting tool projects future daily counts for a selected category or location and shows an uncertainty range.
        - Use these outputs to spot trends and plan resources; remember: models are simplified and should be validated before operational use.
        ''')

    st.header('Export & Reproducibility')
    if st.button('Download filtered dataset as CSV'):
        csv = df_filt.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', data=csv, file_name='filtered_crime.csv', mime='text/csv')

    st.markdown('**Notes & assumptions**')
    st.write('This app makes several simplifying assumptions (e.g., simple imputation, automatic one-hot encoding). For production use you should:')
    st.write('- Clean and standardize location/category names')
    st.write('- Engineer better features for classification')
    st.write('- Use more robust forecasting libraries (Prophet, SARIMA) if needed')

    st.markdown('---')
    st.write('Built for quick analysis and exam/demo use. If you want this converted to a multi-page app or packaged with Docker, I can prepare that next.')
