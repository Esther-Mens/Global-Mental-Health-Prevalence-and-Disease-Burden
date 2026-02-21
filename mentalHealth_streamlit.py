# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config("Global Mental Health Dashboard", layout="wide")
st.title("Global Mental Health Dashboard")

@st.cache_data
def load_and_clean():
    d1 = pd.read_csv("1- mental-illnesses-prevalence.csv")
    d2 = pd.read_csv("2- burden-disease-from-each-mental-illness(1).csv")
    d4 = pd.read_csv("4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv")

    d1 = d1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'schizophrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'depression',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'anxiety',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'bipolar',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'eating_disorders'
    })
    d1 = d1[['Entity', 'Year', 'schizophrenia', 'depression', 'anxiety', 'bipolar', 'eating_disorders']]

    d2.columns = d2.columns.str.strip()
    burden_cols = [c for c in d2.columns if c not in ['entity', 'year', 'Code']]
    d2 = d2[['entity', 'year'] + burden_cols]

    df = pd.merge(d1, d2, left_on=['Entity', 'Year'], right_on=['entity', 'year'], how='inner')

    mental_cols = ['schizophrenia', 'depression', 'anxiety', 'bipolar', 'eating_disorders']
    for c in mental_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in burden_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['total_prevalence'] = df[mental_cols].sum(axis=1)
    df['total_burden'] = df[burden_cols].sum(axis=1)

    return df, mental_cols, burden_cols

df, mental_cols, burden_cols = load_and_clean()

# Sidebar controls
st.sidebar.header("Filters")
countries = sorted(df['Entity'].unique())
sel_countries = st.sidebar.multiselect("Countries (filter)", countries, default=countries[:10])

yr_min, yr_max = int(df['Year'].min()), int(df['Year'].max())
sel_years = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

use_plotly = st.sidebar.checkbox("Use Plotly rendering", True)

if st.sidebar.button("Reset filters"):
    sel_countries = countries[:10]
    sel_years = (yr_min, yr_max)

# Helper: filter
def filt(_df: pd.DataFrame) -> pd.DataFrame:
    d = _df[(_df['Year'] >= sel_years[0]) & (_df['Year'] <= sel_years[1])]
    if sel_countries:
        d = d[d['Entity'].isin(sel_countries)]
    return d

dff = filt(df)

st.sidebar.markdown(f"Data loaded: {len(df):,} rows")
st.sidebar.markdown(f"Filtered: {len(dff):,} rows")

st.header("Visualizations")

# ---- Tabs ----
tab_overview, tab_burden, tab_relationships, tab_trends, tab_priority, tab_change = st.tabs([
    "Overview",
    "Burden (DALYs)",
    "Relationships",
    "Trends",
    "Priority & Hotspots",
    "Change Over Time"
])

with tab_overview:
    st.subheader("Top 10 Countries by Average Mental Illness Prevalence")
    top = dff.groupby('Entity')['total_prevalence'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        top.reset_index(),
        x='Entity',
        y='total_prevalence',
        color='total_prevalence',
        color_continuous_scale='Blues',
        labels={'total_prevalence': 'Avg Prevalence'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Ranks countries by average total prevalence (sum of the five disorders) within the selected filters.")

    st.subheader("Global Contribution of Each Mental Disorder")
    g = dff[mental_cols].mean().sort_values(ascending=False)
    fig = px.bar(x=g.index, y=g.values, labels={'x': 'Disorder', 'y': 'Avg Share'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows which disorders contribute most to average prevalence across the selected data.")

with tab_burden:
    st.subheader("Top 10 Countries by Mental Health Disease Burden (DALYs)")
    topb = dff.groupby('Entity')['total_burden'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        topb.reset_index(),
        x='Entity',
        y='total_burden',
        color='total_burden',
        color_continuous_scale='Reds',
        labels={'total_burden': 'Avg DALYs (rate)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Ranks countries by average total mental-health DALY burden across disorders in the filtered range.")

    st.subheader("Average Disease Burden by Mental Disorder")
    label_map = {
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'Depression',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders': 'Anxiety',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia': 'Schizophrenia',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder': 'Bipolar',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders': 'Eating disorders'
    }
    bur_means = dff[burden_cols].mean().rename(index=label_map).sort_values(ascending=False)
    fig = px.bar(x=bur_means.index, y=bur_means.values, labels={'x': 'Disorder', 'y': 'Avg DALYs (rate)'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Compares average DALY burden by disorder (higher = greater health loss).")

with tab_relationships:
    st.subheader("Correlation Between Average Prevalence and Disease Burden by Country")
    avg = dff.groupby('Entity')[['total_prevalence', 'total_burden']].mean().reset_index()
    fig = px.scatter(
        avg,
        x='total_prevalence',
        y='total_burden',
        hover_name='Entity',
        size='total_burden',
        color='total_prevalence',
        labels={'total_prevalence': 'Avg Prevalence', 'total_burden': 'Avg DALYs (rate)'}
    )
    top10 = avg.nlargest(10, 'total_prevalence')
    for _, r in top10.iterrows():
        fig.add_annotation(x=r.total_prevalence, y=r.total_burden, text=r.Entity, showarrow=False, yshift=10)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows whether countries with higher prevalence also tend to experience higher DALY burden.")

with tab_trends:
    st.subheader("Global Trend of Total Mental Illness Prevalence Over Years")
    trend = dff.groupby('Year')['total_prevalence'].mean().reset_index()
    fig = px.line(trend, x='Year', y='total_prevalence', markers=True,
                  labels={'total_prevalence': 'Avg Total Prevalence'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tracks how average total prevalence changes over time for the filtered selection.")

    st.subheader("Trends in Mental Disorder Prevalence Over Time")
    td = dff.groupby('Year')[mental_cols].mean().reset_index()
    fig = go.Figure()
    for c in mental_cols:
        fig.add_trace(go.Scatter(x=td['Year'], y=td[c], mode='lines+markers', name=c))
    fig.update_layout(yaxis_title="Avg Prevalence (share of population)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Compares time trends across individual disorders (useful for spotting which conditions are rising/falling).")

    st.subheader("Mental Illness Prevalence Trends in Top 10 Countries")
    top10_entities = dff.groupby('Entity')['total_prevalence'].mean().nlargest(10).index.tolist()
    d = dff[dff['Entity'].isin(top10_entities)]
    fig = px.line(d, x='Year', y='total_prevalence', color='Entity', markers=True,
                  labels={'total_prevalence': 'Total Prevalence'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows prevalence trajectories for the highest-prevalence countries within the current filters.")

with tab_priority:
    st.subheader("Top 10 Countries for Mental Health Pilot Programs (Prevalence × Burden)")
    pc = dff.groupby('Entity').agg({'total_prevalence': 'mean', 'total_burden': 'mean'})
    pc['score'] = pc['total_prevalence'] * pc['total_burden']
    fig = px.bar(
        pc.sort_values('score', ascending=False).head(10).reset_index(),
        x='Entity',
        y='score',
        color='score',
        labels={'score': 'Priority Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A simple prioritization heuristic: countries scoring high on both prevalence and burden rise to the top.")

    st.subheader("Low Prevalence but High Disease Burden Countries")
    avgc = dff.groupby('Entity').agg({'total_prevalence': 'mean', 'total_burden': 'mean'}).reset_index()
    cands = avgc.sort_values(['total_prevalence', 'total_burden'], ascending=[True, False]).head(10)
    fig = px.bar(cands, x='Entity', y='total_burden', color='total_burden',
                 labels={'total_burden': 'Avg DALYs (rate)'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Flags places where burden is high relative to measured prevalence (possible severity, under-diagnosis, or access gaps).")

    st.subheader("Countries Consistently in Top 10 for Mental Illness Prevalence")
    d_rank = dff.copy()
    d_rank['rank'] = d_rank.groupby('Year')['total_prevalence'].rank(ascending=False)
    cons = d_rank[d_rank['rank'] <= 10].groupby('Entity').size().sort_values(ascending=False).head(10)
    fig = px.bar(cons.reset_index(name='years'), x='Entity', y='years', color='years',
                 labels={'years': 'Count of Years in Top 10'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Identifies persistent hotspots by counting how often a country appears in the top 10 over time.")

    st.subheader("Share of Global Mental Health Burden (Top 10 vs Rest)")
    tot = avgc.sort_values('total_burden', ascending=False)
    total_burden_sum = tot['total_burden'].sum()
    top10share = (tot.head(10)['total_burden'].sum() / total_burden_sum) if total_burden_sum else 0
    fig = px.pie(values=[top10share, 1 - top10share], names=['Top 10 Countries', 'Rest of World'], hole=0.3)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows concentration of burden: how much of total DALY burden sits in the top 10 countries (by average burden).")

with tab_change:
    st.subheader("Countries Showing Improvement in Mental Health Prevalence")
    early = dff[dff['Year'] <= dff['Year'].min() + 2].groupby('Entity')['total_prevalence'].mean()
    late = dff[dff['Year'] >= dff['Year'].max() - 2].groupby('Entity')['total_prevalence'].mean()
    impr = (early - late).dropna().sort_values(ascending=False).head(10)
    fig = px.bar(impr.reset_index(name='reduction'), x='Entity', y='reduction', color='reduction',
                 labels={'reduction': 'Reduction (early avg − late avg)'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Highlights countries with the largest drop in total prevalence between the early and late windows in the filtered period.")

st.header("Narrative Insights")
st.markdown("""
This dashboard compares **prevalence** (how common disorders are) and **burden** (DALYs: overall health loss).
Use the tabs to explore: **who ranks highest**, **which disorders drive totals**, **how patterns evolve over time**,
and **which countries might be prioritized** based on combined prevalence and burden.
""")
