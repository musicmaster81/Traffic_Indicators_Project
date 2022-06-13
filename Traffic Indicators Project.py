# For this project, we want to analyze a dataset that contains information on traffic volume for the I-94 expressway
# between Minneapolis and Saint Paul. Some of the questions we would like to answer are:
# "Does the time of day affect the volume of traffic?"
# "Which day of the week is the busiest?"
# "How does weather affect the volume of traffic?"

# The dataset documentation mentions that a station located approximately midway between Minneapolis and Saint Paul
# recorded the traffic data. Also, the station only records westbound traffic (cars moving from east to west).
# This means that the results of our analysis will be about the westbound traffic in the proximity of that station.
# In other words, we should avoid generalizing our results for the entire I-94 highway

import pandas as pd
import matplotlib.pyplot as plt

# First, as always, we define the file path for the location of our dataset.
file_path = 'C:\Python\Data Sets\Metro_Interstate_Traffic_Volume.csv'

# Next, we read in our dataset into a dataframe via the read_csv() method.
traffic = pd.read_csv(file_path)
print("Data Frame Information")
print(traffic.info())
print("\n")

# We then would like to begin to explore the dataset to get a general sense of the layout and the data that is recorded.
print("Beginning and end of our data set")
print(traffic.head(10))
print("\n")
print(traffic.tail())
print("\n")

# Let's take a look at what the distribution of the traffic volume looks like. Is it more normal or uniform?
traffic['traffic_volume'].plot.hist()
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')
plt.title('Traffic Volume Histogram')
plt.show()

# It would appear that, on the surface, our distribution is closer to normal than uniform. However, let's look at some
# statistics about the volume of traffic in our dataset.
traffic['traffic_volume'].describe()
print("\n")

# After reviewing these statistics, perhaps a good question to raise is,"Does the time of day skew our data in one way
# or another? As a result, let's begin to isolate the time of day.
traffic['date_time'] = pd.to_datetime(traffic['date_time'])  # Convert the date_time column to a date time object.

# Let's separate the data into 2 different dataframes. One for 7am-7pm and 7pm-7am.
day = traffic.copy()[(traffic['date_time'].dt.hour >= 7) & (traffic['date_time'].dt.hour < 19)]
night = traffic.copy()[(traffic['date_time'].dt.hour >= 19) | (traffic['date_time'].dt.hour < 7)]

# Let's compare the day and night dataframes via a grid chart.
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.hist(day['traffic_volume'])
plt.xlabel('Volume of Traffic')
plt.ylabel('Frequency of Vehicles')
plt.xlim(-50, 8000)
plt.ylim(0, 8000)
plt.title('Traffic from 7am to 7pm')
plt.subplot(1, 2, 2)
plt.hist(night['traffic_volume'])
plt.xlabel('Volume of Traffic')
plt.ylabel('Frequency of Vehicles')
plt.xlim(-50, 8000)
plt.ylim(0, 8000)
plt.title('Traffic from 7pm to 7am')
plt.show()

# Before we draw some conclusions, let's get some insight into the day and night statistics.
day['traffic_volume'].describe()
print("\n")
night['traffic_volume'].describe()
print("\n")

# The statistics and histogram for the nightime dataframe confirms that this dataset is left-skewed, impying that the
# volume of traffic is very light for overnight hours. Our daytime dataframe on the other hand, appears to be normally
# distributed. Let's drill down on this dataset further to obtain more insights.

# Let's isolate the traffic volume for each month. Are there months on average that are more busy than others?
day['month'] = day['date_time'].dt.month
avg_vol_each_month = day.groupby('month').mean()  # Create a new dataframe by computing the average for each month
print("Average Traffic Volume per Month: ", avg_vol_each_month['traffic_volume'])
print("\n")
# It appears that traffic is a lot lighter during the winter months. Let's create a graph to visualize.
plt.plot(avg_vol_each_month['traffic_volume'])
plt.xlabel('Month of the Year')
plt.ylabel('Volume of Traffic')
plt.title('Traffic Volume per Month')
plt.show()

# Our hypothesis is correct, however, there is a very peculiar issue with the data recorded in July. Was there a year
# that is skewing July data?
day['year'] = day['date_time'].dt.year
month_of_july = day[day['month'] == 7]
avg_vol_in_july = month_of_july.groupby('year').mean()
plt.plot(avg_vol_in_july['traffic_volume'])
plt.xlabel('July Year')
plt.ylabel('Frequency')
plt.title('July Traffic Trends')
plt.show()

# It appears that there was a deep drop-off in traffic volume during July in 2016. This is most likely due to construc-
# tion projects as summer months are when these projects are performed. Taking this hypothesis into account, we can con-
# clude the summer months are busier than the winter months.

# Next, let's shift our focus to the day of the week. Is there a day of the week when traffic is busier?
day['day_of_week'] = day['date_time'].dt.dayofweek
avg_per_day = day.groupby('day_of_week').mean()
print(avg_per_day['traffic_volume'])
print("\n")

# Let's plot a line chart to visualize our statistics.
plt.plot(avg_per_day['traffic_volume'])
plt.xlabel('Day of the Week')
plt.ylabel('Volume of Traffic')
plt.title('Volume of Traffic per Day')
plt.show()

# It appears that there is a drop-off in traffic during the weekends. Let's split the dataset into 2: one data frame
# for weekdays and another one for weekends.
day['time_of_day'] = day['date_time'].dt.hour
weekdays = day.copy()[day['day_of_week'] <= 4]
weekends = day.copy()[day['day_of_week'] >= 5]
weekday_hours = weekdays.groupby('time_of_day').mean()
weekend_hours = weekends.groupby('time_of_day').mean()
print("Weekend Daytime Traffic Volume Distribution")
print(weekend_hours['traffic_volume'])
print("\n")
print("Weekday Daytime Traffic Volume Distribution")
print(weekday_hours['traffic_volume'])
print("\n")

# Again, similar to before, let's plot a grid chart to compare the two dataframes.
plt.figure(figsize=(5, 10))
plt.subplot(2, 1, 1)
plt.plot(weekday_hours['traffic_volume'])
plt.xlabel('Time of Day')
plt.ylabel('Traffic Volume')
plt.xlim(6, 19)
plt.ylim(1500, 6500)
plt.title('Average Traffic Volume during Business Days')
plt.subplot(2, 1, 2)
plt.plot(weekend_hours['traffic_volume'])
plt.xlabel('Time of Day')
plt.ylabel('Traffic Volume')
plt.xlim(6, 19)
plt.ylim(1500, 6500)
plt.title('Average Traffic Volume during Weekends')
plt.show()

# We can conclude that weekday mornings are extremely busier than weekend mornings. Weekend travel times appear to me
# more logarithmic as the time of day progresses, there travel times during the week are heaviest at 7am and 3:30pm.

# Lastly, let's see if there is any correlation between the type of weather and traffic volume. We'll leverage the 4
# numerical weather columns from our daytime dataset.
print("Correlation between traffic volume and temperature: ", day['traffic_volume'].corr(day['temp']))
print("\n")
print("Correlation between traffic volume and rain: ", day['traffic_volume'].corr(day['rain_1h']))
print("\n")
print("Correlation between traffic volume and snow: ", day['traffic_volume'].corr(day['snow_1h']))
print("\n")
print("Correlation between traffic volume and cloud coverage: ", day['traffic_volume'].corr(day['clouds_all']))

# Interestingly enough, the only weather with a weak correlation to traffic volume is temperature. Let's construct a
# scatter plot to see if we can perform some more analysis.
day.plot.scatter('traffic_volume', 'temp')
plt.xlabel('Volume of Traffic')
plt.ylabel('Temperature')
plt.title('Temperature vs. Volume')
plt.ylim(200, 325)
plt.show()

# Unfortunately, there isn't a clear-cut trend that we can visually identify for this correlation, which makes sense
# that we came to a weak pearson's "r" value. Let's plot a bar chart to see what the traffic volume looks like on
# average for each weather category.
by_weather_main = day.groupby('weather_main').mean()
by_weather_main['traffic_volume'].plot.barh()
plt.xlabel('Traffic Volume')
plt.ylabel('Type of Weather')
plt.title('Impact of Weather Type on Traffic Volume')
plt.show()

# Again, unfortunately, no weather category gives us a clear-cut answer. Let's break it out by weather description.
plt.figure(figsize=(6, 12))
by_weather_description = day.groupby('weather_description').mean()
by_weather_description['traffic_volume'].plot.barh()
plt.xlabel('Traffic Volume')
plt.ylabel('Description')
plt.title('In-depth Description of Weather')
plt.show()

# Now we come to a good conclusion. Light rain/snow and snow showers impact our traffic volumes substantially.
# In summary, should you want to experience light traffic volumes, travel on weekend mornings when there is no chance of
# snow.
