import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format

df_data = pd.read_csv('data/nobel_prize_data.csv')

def run_challenge(description, func):
    print(f"\nChallenge: {description}")
    input("Press ENTER to see the result...\n")
    func()

# Challenge 1: Print shape, columns, min and max year, head and tail of dataset
def dataset_overview():
    print("Shape of DataFrame:", df_data.shape)
    print("Column names:", df_data.columns.tolist())
    print("First year Nobel Prize awarded:", df_data['year'].min())
    print("Latest year in dataset:", df_data['year'].max())
    print("\nFirst 5 rows:")
    print(df_data.head())
    print("\nLast 5 rows:")
    print(df_data.tail())

# Challenge 2: Check duplicates and NaNs in dataset and show rows with NaN birth_date or organization_name
def data_quality_checks():
    print(f'Any duplicates? {df_data.duplicated().values.any()}')
    print(f'Any NaN values among the data? {df_data.isna().values.any()}')
    print("\nNaN count per column:")
    print(df_data.isna().sum())
    print("\nRows with NaN birth_date (likely organizations):")
    col_subset = ['year','category', 'laureate_type', 'birth_date', 'full_name', 'organization_name']
    print(df_data.loc[df_data.birth_date.isna()][col_subset])
    print("\nRows with NaN organization_name:")
    col_subset_org = ['year','category', 'laureate_type','full_name', 'organization_name']
    print(df_data.loc[df_data.organization_name.isna()][col_subset_org])

# Challenge 3: Convert birth_date to datetime and add prize share percentage column
def preprocess_data():
    df_data['birth_date'] = pd.to_datetime(df_data['birth_date'], errors='coerce')
    separated_values = df_data.prize_share.str.split('/', expand=True)
    numerator = pd.to_numeric(separated_values[0])
    denominator = pd.to_numeric(separated_values[1])
    df_data['share_pct'] = numerator / denominator
    print("\nDataFrame info after conversions:")
    print(df_data.info())

# Challenge 4: Pie chart showing male vs female winners
def plot_gender_distribution():
    biology = df_data.sex.value_counts()
    fig = px.pie(labels=biology.index,
                 values=biology.values,
                 title="Percentage of Male vs. Female Winners",
                 names=biology.index,
                 hole=0.4)
    fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
    fig.show()

# Challenge 5: Show first 3 female Nobel laureates
def first_female_laureates():
    print("\nFirst 3 female laureates:")
    print(df_data[df_data.sex == 'Female'].sort_values('year').head(3)[
        ['full_name', 'year', 'category', 'birth_country', 'organization_name']])

# Challenge 6: Find and show laureates with multiple Nobel Prizes
def multiple_nobel_laureates():
    is_winner = df_data.duplicated(subset=['full_name'], keep=False)
    multiple_winners = df_data[is_winner]
    print(f"\nNumber of laureates awarded Nobel Prize more than once: {multiple_winners.full_name.nunique()}")
    print(multiple_winners[['year', 'category', 'laureate_type', 'full_name']].sort_values(['full_name','year']))

# Challenge 7: Count prizes per category and plot bar chart
def plot_prizes_per_category():
    print(f"\nNumber of categories: {df_data.category.nunique()}")
    prizes_per_category = df_data.category.value_counts()
    v_bar = px.bar(
        x=prizes_per_category.index,
        y=prizes_per_category.values,
        color=prizes_per_category.values,
        color_continuous_scale='Aggrnyl',
        title='Number of Prizes Awarded per Category'
    )
    v_bar.update_layout(xaxis_title='Nobel Prize Category', yaxis_title='Number of Prizes', coloraxis_showscale=False)
    v_bar.show()

# Challenge 8: Show first prizes in Economics category
def first_economics_prizes():
    print("\nFirst prizes in Economics category:")
    print(df_data[df_data.category == 'Economics'].sort_values('year').head(3)[
        ['year', 'full_name', 'prize']])

# Challenge 9: Bar chart for prizes by men vs women per category
def plot_gender_split_per_category():
    cat_men_women = df_data.groupby(['category', 'sex'], as_index=False).agg({'prize': 'count'})
    cat_men_women.sort_values('prize', ascending=False, inplace=True)
    v_bar_split = px.bar(
        x=cat_men_women.category,
        y=cat_men_women.prize,
        color=cat_men_women.sex,
        title='Number of Prizes Awarded per Category split by Men and Women'
    )
    v_bar_split.update_layout(xaxis_title='Nobel Prize Category', yaxis_title='Number of Prizes')
    v_bar_split.show()

# Challenge 10: Plot number of prizes awarded per year with 5 year rolling average
def plot_prizes_per_year():
    prize_per_year = df_data.groupby('year').count()['prize']
    moving_average = prize_per_year.rolling(window=5).mean()
    plt.figure(figsize=(16,8), dpi=200)
    plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(ticks=np.arange(1900, 2021, 5), fontsize=14, rotation=45)
    ax = plt.gca()
    ax.set_xlim(1900, 2020)
    ax.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
    ax.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
    plt.show()

# Challenge 11: Plot number of prizes per year with 5-year moving average and prize share trend with inverted y-axis
def plot_prizes_and_share_trend():
    prize_per_year = df_data.groupby('year').count()['prize']
    moving_average = prize_per_year.rolling(window=5).mean()
    yearly_avg_share = df_data.groupby('year').agg({'share_pct': 'mean'})
    share_moving_average = yearly_avg_share.rolling(window=5).mean()
    plt.figure(figsize=(16,8), dpi=200)
    plt.title('Number of Nobel Prizes Awarded per Year with Prize Share', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(ticks=np.arange(1900, 2021, 5), fontsize=14, rotation=45)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_xlim(1900, 2020)
    ax1.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
    ax1.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
    ax2.plot(prize_per_year.index, share_moving_average.values, c='grey', linewidth=3)
    ax2.invert_yaxis()
    plt.show()

# Challenge 12: Plot horizontal bar chart of top 20 countries by total prizes
def plot_top20_countries():
    top_countries = df_data.groupby('birth_country_current', as_index=False).agg({'prize': 'count'})
    top_countries.sort_values(by='prize', inplace=True)
    top20_countries = top_countries[-20:]
    h_bar = px.bar(
        x=top20_countries.prize,
        y=top20_countries.birth_country_current,
        orientation='h',
        color=top20_countries.prize,
        color_continuous_scale='Viridis',
        title='Top 20 Countries by Number of Prizes'
    )
    h_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Country', coloraxis_showscale=False)
    h_bar.show()

# Challenge 13: Plot choropleth map of prizes by country
def plot_choropleth_prizes_by_country():
    df_countries = df_data.groupby(['birth_country_current', 'ISO'], as_index=False).agg({'prize': 'count'})
    world_map = px.choropleth(
        df_countries,
        locations='ISO',
        color='prize',
        hover_name='birth_country_current',
        color_continuous_scale=px.colors.sequential.matter
    )
    world_map.update_layout(coloraxis_showscale=True)
    world_map.show()

# Challenge 14: Stacked horizontal bar chart of top 20 countries by prizes split by category
def plot_top20_countries_by_category():
    cat_country = df_data.groupby(['birth_country_current', 'category'], as_index=False).agg({'prize': 'count'})
    cat_country.sort_values(by='prize', inplace=True)
    top_countries = df_data.groupby('birth_country_current', as_index=False).agg({'prize': 'count'})
    top_countries.sort_values(by='prize', inplace=True)
    top20_countries = top_countries[-20:]
    merged_df = pd.merge(cat_country, top20_countries, on='birth_country_current')
    merged_df.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
    merged_df.sort_values(by='total_prize', inplace=True)
    cat_cntry_bar = px.bar(
        x=merged_df.cat_prize,
        y=merged_df.birth_country_current,
        color=merged_df.category,
        orientation='h',
        title='Top 20 Countries by Number of Prizes and Category'
    )
    cat_cntry_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Country')
    cat_cntry_bar.show()

# Challenge 15: Line chart of cumulative prizes over time by country
def plot_cumulative_prizes_over_time():
    prize_by_year = df_data.groupby(['birth_country_current', 'year'], as_index=False).count()
    prize_by_year = prize_by_year.sort_values('year')[['year', 'birth_country_current', 'prize']]
    cumulative_prizes = prize_by_year.groupby(['birth_country_current', 'year']).sum().groupby(level=0).cumsum()
    cumulative_prizes.reset_index(inplace=True)
    l_chart = px.line(
        cumulative_prizes,
        x='year',
        y='prize',
        color='birth_country_current',
        hover_name='birth_country_current'
    )
    l_chart.update_layout(xaxis_title='Year', yaxis_title='Number of Prizes')
    l_chart.show()

# Challenge 16: Print top 20 organizations by number of prizes
def print_top20_organizations():
    top20_orgs = df_data.organization_name.value_counts().head(20)
    print("\nTop 20 Organizations by Number of Prizes:")
    print(top20_orgs)

# ------------------ Run Challenges ------------------

run_challenge("Dataset overview: shape, columns, min/max year, head and tail", dataset_overview)

run_challenge("Data quality checks: duplicates and NaNs", data_quality_checks)

run_challenge("Preprocess data: birth_date datetime and prize share %", preprocess_data)

run_challenge("Plot gender distribution (Male vs Female winners)", plot_gender_distribution)

run_challenge("Show first 3 female Nobel laureates", first_female_laureates)

run_challenge("Find laureates with multiple Nobel Prizes", multiple_nobel_laureates)

run_challenge("Plot number of prizes awarded per category", plot_prizes_per_category)

run_challenge("Show first prizes in Economics category", first_economics_prizes)

run_challenge("Plot prizes by men vs women per category", plot_gender_split_per_category)

run_challenge("Plot number of prizes awarded per year with rolling average", plot_prizes_per_year)

run_challenge("Plot prizes per year and prize share trend", plot_prizes_and_share_trend)

run_challenge("Plot top 20 countries by total prizes", plot_top20_countries)

run_challenge("Plot choropleth map of prizes by country", plot_choropleth_prizes_by_country)

run_challenge("Plot stacked bar chart of top 20 countries by category", plot_top20_countries_by_category)

run_challenge("Plot cumulative prizes over time by country", plot_cumulative_prizes_over_time)

run_challenge("Print top 20 organizations by number of prizes", print_top20_organizations)
