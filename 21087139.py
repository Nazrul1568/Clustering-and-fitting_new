#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
pd.options.mode.chained_assignment = None


# In[2]:


def load_csv_data(filename):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    ------------    
    filename (str): The filename of the CSV file to be loaded.

    Returns:
    ---------    
    data_frame (pandas.DataFrame): The DataFrame containing the data 
    read from the CSV file.
    """
    file_path = filename
    print(file_path)
    data_frame = pd.read_csv(file_path, skiprows=4)
    data_frame = data_frame.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 67'])
    return data_frame


# In[3]:


def merge_indicators(country_data_frame1, country_data_frame2, indicator_name1, indicator_name2, year):
    """
    Merge two data frames based on the 'Country Name' and a specific year, keeping only the rows
    where data is available for both indicators. The data frames should contain country-specific 
    data for different indicators.

    Parameters:
    - country_data_frame1 (DataFrame): The first data frame containing country names and values 
                                       for the first indicator.
    - country_data_frame2 (DataFrame): The second data frame containing country names and values 
                                       for the second indicator.
    - indicator_name1 (str): The name of the first indicator to be used as the column name in the 
                             merged data frame.
    - indicator_name2 (str): The name of the second indicator to be used as the column name in the 
                             merged data frame.
    - year (str): The year for which the data is being merged. It should be a column name in both 
                  data frames.

    Returns:
    - DataFrame: A new data frame with two columns: one for each indicator, renamed to the 
                  specified indicator names, containing data from countries where both indicators 
                  are available.
    """
    country_data_frame1 = country_data_frame1[['Country Name', year]]
    country_data_frame2 = country_data_frame2[['Country Name', year]]
    
    merged_data_frame = pd.merge(country_data_frame1, country_data_frame2,
                                 on="Country Name", how="outer")
    merged_data_frame.dropna(inplace=True)
    merged_data_frame.rename(columns={year + "_x": indicator_name1, year + "_y": indicator_name2}, inplace=True)
    
    indicators_data_frame = merged_data_frame[[indicator_name1, indicator_name2]].copy()
    return indicators_data_frame


# In[4]:


def label_countries(indicator_name1, indicator_name2, data_frame1, data_frame2, year):
    """
    Merge two data frames based on the 'Country Name' for a given year, including data for two specified 
    indicators. The function ensures data is present for both indicators before including a country in the result.

    Parameters:
    - indicator_name1 (str): Name of the first indicator, which will serve as a column name in the output.
    - indicator_name2 (str): Name of the second indicator, which will also serve as a column name in the output.
    - data_frame1 (DataFrame): First data frame containing 'Country Name' and data for the first indicator of the specified year.
    - data_frame2 (DataFrame): Second data frame containing 'Country Name' and data for the second indicator of the specified year.
    - year (str): The year column in both data frames that contains the indicator values.

    Returns:
    - DataFrame: A new data frame that includes the 'Country Name', and the two indicators, ensuring that
                 data exists for both indicators before including any country.
    """
    data_frame1 = data_frame1[['Country Name', year]]
    data_frame2 = data_frame2[['Country Name', year]]
    
    merged_data_frame = pd.merge(data_frame1, data_frame2, on="Country Name", how="outer")
    merged_data_frame.dropna(inplace=True)
    merged_data_frame.rename(columns={year + "_x": indicator_name1, year + "_y": indicator_name2}, inplace=True)
    
    labeled_countries_data_frame = merged_data_frame[['Country Name', indicator_name1, indicator_name2]].copy()
    return labeled_countries_data_frame


# In[5]:


def logistic_function(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


# In[6]:


def fit_and_predict(data_frame, country_name, indicator, title, title_forecast, initial_guess):
    """
    Fit logistic model to the historical data of a given country and indicator, and predict future values,
    including a forecast plot with error propagation.

    Parameters:
    - data_frame (DataFrame): The data frame containing the time series data.
    - country_name (str): The name of the country for which the model is fitted and forecasted.
    - indicator (str): The indicator being modeled and forecasted.
    - title (str): Title for the plot of the historical data fit.
    - title_forecast (str): Title for the forecast plot.
    - initial_guess (list): Initial guess for the parameters of the logistic function.

    Outputs:
    - Saves two plots to the filesystem: one showing the logistic fit and another showing the forecast with error bounds.
    """
    # Fitting the model to the data
    popt, pcorr = opt.curve_fit(logistic_function, data_frame.index, data_frame[country_name], p0=initial_guess)
    data_frame["pop_log"] = logistic_function(data_frame.index, *popt)

    # Plotting the fit to historical data
    plt.figure()
    plt.plot(data_frame.index, data_frame[country_name], label="data", color='navy')
    plt.plot(data_frame.index, data_frame["pop_log"], label="fit", color='brown')
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel(indicator)
    plt.title(title)
    plt.savefig(f'{country_name}_fit.png', dpi=300)

    # Forecasting future values
    years = np.linspace(1995, 2030)
    forecast_values = logistic_function(years, *popt)
    sigma = err.error_prop(years, logistic_function, popt, pcorr)
    lower_bound = forecast_values - sigma
    upper_bound = forecast_values + sigma

    # Plotting the forecast with error bands
    plt.figure()
    plt.title(title_forecast)
    plt.plot(data_frame.index, data_frame[country_name], label="data", color='navy')
    plt.plot(years, forecast_values, label="Forecast")
    plt.fill_between(years, lower_bound, upper_bound, alpha=0.5, color="brown")
    plt.legend(loc="upper left")
    plt.xlabel('Years')
    plt.ylabel(indicator)
    plt.savefig(f'{country_name}_forecast.png', dpi=300)
    plt.show()


# In[7]:


def extract_country_data(data_frame, country_name, start_year, end_year):
    """
    Extract time series data for a specific country and range of years from a DataFrame.

    Parameters:
    - data_frame (DataFrame): DataFrame containing multiple countries' data, indexed by years
                              in the first row and country names in the first column.
    - country_name (str): The name of the country for which data is extracted.
    - start_year (int): The start year of the range for which data is to be extracted.
    - end_year (int): The end year of the range for which data is to be extracted.

    Returns:
    - DataFrame: A DataFrame containing the extracted data for the specified country
                 and range of years, with years as the index.
    """
    # Transpose to switch columns and rows for easier slicing
    transposed_df = data_frame.T

    # Set the first row as column headers
    transposed_df.columns = transposed_df.iloc[0]

    # Remove the row containing the headers from the data
    transposed_df = transposed_df.drop(['Country Name'])

    # Filter the data frame to include only the specified country
    country_data = transposed_df[[country_name]]

    # Convert the index to integer for proper year filtering
    country_data.index = country_data.index.astype(int)

    # Filter data between specified start and end years (inclusive of end year)
    country_data = country_data[(country_data.index > start_year) & (country_data.index <= end_year)]

    # Ensure the data is in float format for any subsequent analysis
    country_data[country_name] = country_data[country_name].astype(float)

    return country_data


# In[8]:


def perform_clustering(data_frame, indicator1, indicator2, x_label, y_label, title, num_cluster_centers, 
                       fitted_data, data_min, data_max):
    """
    Perform KMeans clustering on a subset of data and visualize the results with a scatter plot, 
    including the rescaled cluster centers.

    Parameters:
    - data_frame (DataFrame): The main data frame containing the data to be plotted.
    - indicator1 (str): Column name of the first indicator for clustering.
    - indicator2 (str): Column name of the second indicator for clustering.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - title (str): Title of the plot.
    - num_cluster_centers (int): Number of clusters to form.
    - fitted_data (DataFrame): Data used for fitting the KMeans model.
    - data_min (array): Minimum scaling bounds used for rescaling the cluster centers.
    - data_max (array): Maximum scaling bounds used for rescaling the cluster centers.

    Returns:
    - array: Labels of the clusters for each data point in the main data frame.
    """
    # Initialize and fit the KMeans model
    kmeans = cluster.KMeans(n_clusters=num_cluster_centers, n_init=10, random_state=0)
    kmeans.fit(fitted_data)

    # Extract labels and cluster centers
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Plotting
    plt.figure()
    scatter_plot = plt.scatter(data_frame[indicator1], data_frame[indicator2], c=labels, cmap="Paired")

    # Rescale and show cluster centers
    scaled_centers = ct.backscale(cluster_centers, data_min, data_max)
    plt.scatter(scaled_centers[:, 0], scaled_centers[:, 1], c="black", marker="d", s=80)

    # Plot adjustments
    plt.legend(*scatter_plot.legend_elements(), title="Clusters")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()

    return labels


# In[9]:


def plot_silhouette_score(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to evaluate.

    Returns:
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='r')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


# In[10]:


co2_emissions_per_capita = load_csv_data('CO2_emissions_metric_tons_per_capita.csv')
gdp_per_capita = load_csv_data('GDP_per_capita_current_US$.csv')

# Get data for clustering
cluster_data = merge_indicators(gdp_per_capita, co2_emissions_per_capita,'GDP_per_capita_current_US$', 'CO2_emissions_metric_tons_per_capita', '2020')

# Scaling the cluster data
scaled_data, data_min_values, data_max_values = ct.scaler(cluster_data)
plot_silhouette_score(scaled_data, 12)

# Perform clustering
cluster_labels = perform_clustering(cluster_data, 'GDP_per_capita_current_US$', 'CO2_emissions_metric_tons_per_capita', 'GDP per capita current US$', 'CO2 emissions metric tons per capita', 'CO2 emissions metric tons vs GDP current US$ (per capita) in 2020', 3, scaled_data, data_min_values, data_max_values)

# Label countries and merge cluster labels
labeled_countries = label_countries('GDP_per_capita_current_US$', 'CO2_emissions_metric_tons_per_capita', gdp_per_capita, co2_emissions_per_capita, '2020')
labeled_countries['cluster_label'] = cluster_labels
selected_countries = labeled_countries[labeled_countries['Country Name'].isin(['United Kingdom', 'United States'])]

# Get and prepare data for the United States
us_gdp_data = extract_country_data(gdp_per_capita, 'United States', 1990, 2020)
us_gdp_data = us_gdp_data.fillna(0)
fit_and_predict(us_gdp_data, 'United States', 'GDP_per_capita_current_US$', "GDP per Capita Current US$ in United States 1990-2020", "GDP per Capita Current US$ in United States Forecast Until 2030", (1e5, 0.04, 1990))

# Get and prepare data for the United Kingdom
uk_gdp_data = extract_country_data(gdp_per_capita, 'United Kingdom', 1990, 2020)
uk_gdp_data = uk_gdp_data.fillna(0)
fit_and_predict(uk_gdp_data, 'United Kingdom', 'GDP_per_capita_current_US$', "GDP per Capita Current US$ in United Kingdom 1990-2020", "GDP per Capita Current US$ in United Kingdom Forecast Until 2030", (1e5, 0.04, 1990))

# Get and prepare CO2 emissions data for the United States
us_co2_data = extract_country_data(co2_emissions_per_capita, 'United States', 1990, 2020)
us_co2_data = us_co2_data.fillna(0)
fit_and_predict(us_co2_data, 'United States', 'CO2_emissions_metric_tons_per_capita', "CO2 Emissions Metric Tons Per Capita in United States 1990-2020", "CO2 Emissions Metric Tons Per Capita in United States Forecast Until 2030", (20, 0.04, 1990))

# Get and prepare CO2 emissions data for the United Kingdom
uk_co2_data = extract_country_data(co2_emissions_per_capita, 'United Kingdom', 1990, 2020)
uk_co2_data = uk_co2_data.fillna(0)
fit_and_predict(uk_co2_data, 'United Kingdom', 'CO2_emissions_metric_tons_per_capita', "CO2 Emissions Metric Tons Per Capita in United Kingdom 1990-2020", "CO2 Emissions Metric Tons Per Capita in United Kingdom Forecast Until 2030", (10, 0.04, 1990))


# In[ ]:




