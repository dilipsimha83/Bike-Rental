# Bike-Rental











Project Report 
on 
Bike Renting
by
Dilip Simha






Contents

1.	Chapter1: Introduction

1.1	Problem Statement…………………………………………………………………………………….3

2.	Chapter2: Methodology

a.	Pre-Processing………………………………………………………………………………………5

b.	Exploratory Data Analysis…………………………………………………………………….5

c.	Visualisations using Boxplot    ……………………………………………………………   5

d.	Correlation ………………………………………………………………………………………….9

e.	histogram of numerical variables …………………………………………………………9

f.	Missing Value and Outlier…………………………………………………………………… 10

3.	Chapter3: Model Selection

a.	Decision Tree ………………………………………………………………………………………11

b.	Linear Regression ………………………………………………………………………………  12

4.	Chapter4: Model Evaluations

a.	R Square……………………………………………………………………………………………… 13

b.	Adjusted R Square…………………………………………………………………………………14

c.	MAE ………………………………………………………………………………………………………14

d.	RMSE …………………………………………………………………………………………………….14

e.	MAPE…………………………………………………………………………………………………….14



5.	Chapter5: Conclusion………………………………………………………………………………………….15



Chapter1: Introduction
Problem statement    
The objective of this Case is to Predication of bike rental count on daily based on the environmental and seasonal settings. 
About ride rental and sharing
Ride sharing companies like Uber, Ola are great business models that provide convenient, affordable and efficient transportation options for customers who want to go to places without the hassle of owning or operating a vehicle. However, with the increasing number of automobiles, riding sharing in cars are not efficient enough especially in crowded and busy areas like cities' downtown. Therefore, bike sharing is a brilliant idea which provides people with another short-range transportation option that allows them to travel without worrying about being stuck in traffic and maybe enjoy city view or even workout at the same. 
About Data:
The dataset is provided Edwisor. The data contains 731 observations and 18 variables.
The variables are:
•	"datetime", containing hourly date in timestamp format;
•	"season", containing integers 1 to 4 representing "Winter", "Spring","Summer","Fall";
•	"holiday", containing Boolean expressions in 1s and 0s representing whether the day of the observation is a holiday or not;
•	"workingday", containing Boolean expressions in 1s and 0s representing whether the day of the observation is a working day or not;
•	"weather", containing integers 1 to 4 representing four different lists of weather conditions:
1: Clear or cloudy,
2: Mists,
3: Light rain or snow,
4: Heavy rain, snow or even worse weather.
•	"temp", containing values of temperature at the given time;
•	"atemp", containing values of feeling temperature at the given time;
•	"humidity", containing values of relative humidity level at the given time, in the scale of 1 to 100;
•	"windspeed", containing values of wind speed, in mph (miles per hour);

•	"casual", containing the count of non-registered user rental;
•	"registered", containing the count of registered user rentals;
•	"cnt", containing the total count of rentals at the given hour, across all stations
•	
"cnt" will be used as response variable here, and all other as predictor.
However in decision tree methodology I have used “ registered” too as a dependent variable in one analysis and “cnt” in another analysis.





















Chapter 2: Methodology
Pre-Processing
Data is fairly clean and well organized.  There are very few/no columns with missing data or default data that needs to be dealt with.  So, there is very little feature imputation required. 
Exploratory Data Analysis
Without constructing any models, let us explore the dataset.
I have proceeded to construct a data frame that summarizes data based on season, day of the week, temperature, humidity, windspeed, casual, registered users of the bike.
The purpose of this summarization is to find a general relationship between variables regardless of which year the data is from (since the data spans two years and the business is growing.)
Visualisation
Using the summarized data frame, we can visualize some of the features of the data without looking at a complex summary statistics.

A. Season Vs Hiring

 
Here we are seeing the boxplot of weather vs registered users. 

Let us see another boxplot with the total count versus weather
 
In the above plot it can be seen that the season or weather is most certainly having an impact on the hiring of bike.
This is certainly one of points in the hypothesis that weather is having an impact on hiring.
The boxplot of different seasons against bike rental count reveals that there is a seasonal trend with the rental count. Rental count is generally low or nil in Winter and it peaks in Summer. Season can be one of the determining factors that affects bike rental count

B. Days of the week Vs Hiring
The assumption here is verify if days of the week have an impact on hiring.
The days have been extracted from the date stamp variable.
 
The plot reflects a normal distribution of hiring on all the days of the week, particularly the hiring is higher on Saturday.  This could be due to the fact that many events organised promoting people to use bikes to save on parking and congestion in malls. Here the term bike is bi cycles as per my assumption since the given data does not specify if it is fuel driven or electric driven one.
C. Temperature Vs Hiring
One other hypothesis that we can test is to know if temperature is having an impact on the hiring.
 
The temperature plot shows that generally, the warmer the temperature, the higher bike rental demand. 

D. Humidity Vs Hiring

 
The humidity plot shows that generally, the higher the relative humidity, the lower bike rental demand. 

E. Wind speed Vs Hiring
 
The plot reveals that people may enjoy a mild wind speed, lower the wind speed higher is the rental hiring, also higher the windspeed lower is the hiring. This could be majorly due to the weather condition. Windspeeds could be higher during rainy and snow clad conditions.



Correlation
         	b. temp	b.atemp	b.hum	b.windspeed
b. temp	1.0000000	0.9917016	0.1269629	-0.1579441
b.atemp	0.9917016	1.0000000	0.1399881	-0.1836430
b.hum	0.1269629	0.1399881	1.0000000	-0.2484891
b.windspeed	-0.1579441	-0.1836430	-0.2484891	1.0000000

The above tables reflect the values obtained by performing a correlation code.
The correlation function helps us to know or view what variables are strongly correlated and what is weakly correlated.
The above tables clearly shows that the variables are not strongly correlated.
This helps us in our above assumptions and hypothesis which is clearly visible in the plots.
Here, we find that temperature is inversely corelated i.e as temperature increases the windspeed falls. So also, as humidity increases windspeed falls.
Temperature is closely related to humidity as the latter is a by product of the former. Hence the values are positively correlated but do not disturb the assumptions.
Understanding the distribution of numerical variables
 
From the above histogram plotted for the data provided even before any modifications were implemented it can be seen that season is equally distributed, data occurs max during weather situation1 which is clear sky, partial clouds, humidity windspeed and temperature are normally distributed. Looking into whether hiring is maximum during working day or holiday or weekend, data shows a high hiring during working days.

Missing Value and Outliers
The data provided has no missing values and outliers which could be found in the given dataset.  This was verified using the necessary code. Hence the methods of imputation and outlier treatment has not been done.

 









Chapter 3: Model Selection

Decision Tree
Decision Tree algorithm belongs to the family of supervised learning algorithms. Decision trees are used for both classification and regression problems. A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome (categorical or continues value).  The general motive of using Decision Tree is to create a training model which can use to predict class or value of target variables by learning decision rules inferred from prior data (training data).
 
The decision tree above is generated having “cnt” as the dependent variable. It can be very clearly seen that the model has split mainly based on registered and casual and no other variable.  Another model was worked out keeping registered as the dependent variable.
 
Here the parent node is “cnt” which is spilt into tree taking working day and casual users.
Linear Regression Model
Linear regression is used for finding linear relationship between target and one or more predictors. There are two types of linear regression- Simple and Multiple.
In the case of simple linear regression, one is the predictor or target variable and the other is the independent variable. It is useful to find out the relation between two continuous variables. One example for this is the relation between height and weight, speed and distance etc.
The core idea is to obtain a line that best fits the data. The best fit is line is the one for which the total prediction error is as small as possible. Error is the distance between the point to the regression line
In the case of multiple linear regression, it explains the relation between one continuous dependent variable and multiple independent variables.
 
The table shows the summary of the p-values obtained after running the linear regression model.
Hypothesis: 
H0(null hypothesis): the variables have an impact on the bike rental
H1(alternate hypothesis): the variables do not have any impact on bike rental, any variation is simply due to random variation.
Any regression model should have these hypothesis, as the ultimate aim is to prove if the we are accepting the null hypothesis or the alternate hypothesis depending on the p-value and acceptable threshold value or cut -off value range.
Assuming that threshold value is 0.05, we find that season, temperature, windspeed, humidity emerges as most important variables that have an impact on bike rental.
Hence, we reject the alternate hypothesis and accept the null hypothesis.
Chapter 4: Model Evaluation
R squared
R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.
The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model. Or:
R-squared = Explained variation / Total variation
R-squared is always between 0 and 100%:
•	0% indicates that the model explains none of the variability of the response data around its mean.
•	100% indicates that the model explains all the variability of the response data around its mean.
 
In my model, I have got a R-squared which is 0.5413 which is an agreeable value though not a great value. Higher R squared is preferred. As higher the R squared results in a better model.




Adjusted R – square
Both R2 and the adjusted R2 give you an idea of how many data points fall within the line of the regression equation. However, there is one main difference between R2 and the adjusted R2: R2 assumes that every single variable explains the variation in the dependent variable. The adjusted R2 tells you the percentage of variation explained by only the independent variables that actually affect the dependent variable.
Here adjusted R squared is 0.5357, which means only approximately 54% of variance is accounted or explained and the remaining 46% has not been explained.
Increasing the variables will certainly increase R squared and this will lead to a better fit, however it can be misleading as this may lead to over fitting and can lead to misleading projections.
Error Metrics
MAE (Mean Absolute Error)
MAE is the simplest error metric to understand. We’ll calculate the residual for every data point, taking only the absolute value of each so that negative and positive residuals do not cancel out. We then take the average of all these residuals. Effectively, MAE describes the typical magnitude of the residuals.
RMSE (Root Mean Square Error)
Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around line of best fit. 
MAPE (Mean Absolute Percentage Error)
The mean absolute percentage error (MAPE) is a statistical measure of how accurate a forecast system is. It measures this accuracy as a percentage.
 

Chapter 5: Conclusion
MAE value is 1164.77 which means that my model is off by 1164.77.
RMSE value is 1371.53. RMSE unit is same as dependent variable. In our case study, the dependent variable “cnt” is a count of bike rental. The maximum value is 8714, hence this value of RMSE is small and hence acceptable.
MAPE value is 0.3463 which denotes that the model is reasonable in forecasting.
R square and Adjusted R square is 0.54 which is merely an agreeable value.
Keeping in view the case study problem and analysing the dependent variable, detailed data exploration, collinearity test, decision tree, random forest and linear regression model techniques were applied. More importantly I have made the hypothesis assumptions of null and alternate while applying the multiple regression model wherein I have rejected the alternate hypothesis and agreeing to the null hypothesis.
All these prove significantly that the given variables do have an impact on hiring of bike rental. 
The environmental and seasonal factors are very critical for hiring of bike.
The boxplots and histograms have shown that there is normal distribution of hiring during the weekdays significantly higher on Saturdays.
So the company looking to increase the bike rentals could possibly explore a few suggestions:
a.	Increase awareness about the traffic congestion and parking problems encouraging people to start using their bike rentals which could reduce the anxiety and stress amongst people.
b.	Use promotional advertisements to working people particularly in the age group of 20 to 50 by providing the health benefits of cycling and use of bike rentals
c.	Provide attractive gift vouchers when registered members complete certain number of hiring. Referral gifts is also another way to rope in more people
d.	Have tie ups with certain corporate and government companies explaining the benefits both to the company and individuals by talking about the parking infrastructure, health and green benefits. This could be probably even made on 3 working days in a week. This will increase the revenues on working days.
e.	Saturday has higher hiring, however Sunday too is a weekend. Having biking marathons on Sunday for social causes will be another way to increase hiring on Sundays as well.
The above suggestions are made purely to increase the revenue of the bike rental company assuming that it is bi cycle. The above can be made use during appropriate weather conditions and seasons.

R Code

m=ls()
getwd()
setwd("C:/Users/Dilip/Desktop/Edwisor/Project 3 Bike Rental")
b<-read.csv("day.csv")
dim(b)
str(b)
names(b)
str(b$yr)
summary(b$yr)
#### convert dteday as date format#####
b$dteday=as.Date(b$dteday,format="%Y-%m-%d")
str(b)
#### Missing values######

sum(is.na(b))
table(is.na(b))
### There are no missing values.
### understanding the distribution of numeric variables

par(mfrow=c(4,2))
par(mar = rep(2, 4))
hist(b$season) ### has equal distribution
hist(b$weather)#### weather 1 has highest contribution, clear weather
hist(b$hum)
hist(b$holiday)### holidays having maximum demand
hist(b$workingday)### max between working days and weekends.
hist(b$temp)
hist(b$atemp)
hist(b$windspeed)### looks normally distributed

### convert season, weather, working day , holiday as factor variables.

b$season=as.factor(b$season)
b$weather=as.factor(b$weather)
b$holiday=as.factor(b$holiday)
b$workingday=as.factor(b$workingday)
b$weekday=as.factor(b$weekday)

### checking the structure of the dataset

str(b)

### Hypothesis : daily trend having impact on hiring?

date=substr(b$dteday,1,10)
days<-weekdays(as.Date(date))
b$day=days
str(b)
b$day=as.factor(b$day)
library(ggplot2)
box_plot<-ggplot(b,aes(x = day,y = cnt))
box_plot+geom_boxplot()
### It can be seen that hiring is maximum on saturday wednesday and thursday other days being normally distributed.
### We can conclude that daily hiring is having an impact on hiring.

### Let us see the impact with casual hiring########
box_plot<-ggplot(b,aes(x = day,y = casual))
box_plot+geom_boxplot()
### plot clearly shows that casual hiring is max on saturdays and sundays. Noticable outliers on monday friday and wednesday

### Let us see the impact of registered users with days
box_plot<-ggplot(b,aes(x = day,y = registered))
box_plot+geom_boxplot()
### registered users seem to be normally distributed.

### Let us see the impact of weather on registered users hiring
box_plot<-ggplot(b,aes(x = weather,y = registered))
box_plot+geom_boxplot()
#### Let us see the impact of the count vs weather
box_plot<-ggplot(b,aes(x = weather,y = cnt))
box_plot+geom_boxplot()

#### Impact of temperature on hiring

ggplot(b, aes(x = temp, y = cnt, color = weathersit)) +
  geom_smooth(fill = NA, size = 1) +
  theme_light(base_size = 11) +
  xlab("Temperature") +
  ylab("Number of Bike Rentals") +
  ggtitle("\n") +
  scale_color_discrete(name = "Type of Weather:",
                       breaks = c(1, 2, 3, 4),
                       labels = c("Clear or Cloudy", 
                                  "Mist", 
                                  "Light Rain or Snow", 
                                  "")) +
  theme(plot.title = element_text(size = 11, face="bold"))
### the temperature plot shows that higher the temperature, higher is the hiring.

#########################Impact of Humidity on Hiring#################

ggplot(b, aes(x = hum, y = cnt, color = weathersit)) +
  geom_smooth(method = 'loess', fill = NA, size = 1) +
  theme_light(base_size = 11) +
  xlab("Humidity") +
  ylab("Number of Bike Rentals") +
  ggtitle("\n") +
  scale_color_discrete(name = "Type of Weather:",
                       breaks = c(1, 2, 3, 4),
                       labels = c("Clear or Cloudy", 
                                  "Mist", 
                                  "Light Rain or Snow", 
                                  "")) +
  theme(plot.title = element_text(size = 11, face="bold"))
### Higher the humidity lower is the hiring inference from the plot

#####################Impact of Wind speed on Hiring #######################
ggplot(b, aes(x = windspeed, y = cnt)) +
  geom_smooth(fill = NA, size = 1) +
  theme_light(base_size = 11) +
  xlab("Wind Speed") +
  ylab("Number of Bike Rentals") +
  ggtitle("\n") +
  scale_color_discrete(name = "Type of Weather:",
                       breaks = c(1, 2, 3, 4),
                       labels = c("Clear or Cloudy", 
                                  "Mist", 
                                  "Light Rain or Snow", 
                                  "")) +
  theme(plot.title = element_text(size = 11, face="bold"))

summary(b$weather)
###weather:    Four Categories of weather
##1-> Clear, Few clouds, Partly cloudy, Partly cloudy
##2-> Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
##3-> Light Snow and Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
##4-> Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
### from the plot we can see that weathers 1,2 and 3 having maximum hiring of registered users.
### Clearly creiteria 4 which is heavy rain and ice pallets do not have hiring. 
### We can infer that there is max hiring during the clear clouds. There is seems to be hiring during
#### light snow and rainfall too.

#################Let us verify the correlation with the continous variables in the data set#######

sub = data.frame(b$temp,b$atemp,b$hum,b$windspeed)
cor(sub)

### We can infer frm the correlation table that temperature, atemperature and humidity are
### inversely correlated to windspeed.

################################Building Desicion Tree ##################################

set.seed(100)
train_index = sample(1:nrow(b), 0.8*nrow(b))### Simple random sampling method used as target variable is continous        
train = b[train_index,]
test = b[-train_index,]

#### Build Descision Tree using Rpart#####
library(rpart)
library(rpart.plot)

dt_model = rpart(cnt~.,data = train)

### Plot the descision Tree ###

rpart.plot(dt_model)

### the decision tree using cnt as target is mainly suggesting to use casual and registered.

### Let us study what the decision tree suggests if make registered as our dependent variable.

dt_model1 = rpart(registered~.,data = train)

### Let us plot this model to see what variables does the model suggest

rpart.plot(dt_model1)

### the descision tree is indicating to consider cnt, workingday and casual users.

#### Predict for test cases ######

dt_predictions = predict(dt_model1, test[])
dt_predictions

########################Create data frame for actual and predicted values####################

df_pred = data.frame("actual"= test[], "dt_pred"=dt_predictions)
head(df_pred)

#### Confusion Matrix cannot be implemented as the dependent variable is a continous function,
#### confusion matrix is used to evaluate in classification model

##########################Feature Scaling #####################################

qqnorm(train$temp) ####temperature is normally distributed as the graph is nearly straight.
hist(train$cnt)### the count of users is normally distributed assuming this to be our dependent variable.
colnames(train)
cnames<-colnames(train)

#################################### Random Forest ####################################
install.packages("randomForest")
library(randomForest)

RF_model = randomForest(registered ~ ., train[,-18], importance = TRUE, ntree = 100)
#### Note: column 18 is representing the days of the week which is a character variable.
#### randomforest() fucntion threw an error, due to the presence of character variable,hence
### we are dropping the character variable to process random forest.
str(train)
### we need to extract the rules from Random Forest
#transform rf object to an inTrees' format

install.packages("RRF")
library(RRF)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-15])

################################ Linear Regression Model ####################

l_model = lm(registered ~ ., data = train)

############ Summary of the model with registered as dependent variable #############

summary(l_model)


##### Lets see another model #####

l_model1 = lm(cnt~., data = train)
#### lets see the summary of thisl_model1

summary(l_model1)

l_model1step<-step(l_model)

summary(l_model1step)

#### Third model #####

l_model2 = lm(formula = cnt~season+holiday+weekday+windspeed+temp+atemp+hum,data = train)

#### lets see the summary of the third model#####

summary(l_model2)

l_model2step<-step(l_model2)

summary(l_model2step)

### Certainly the second model is better than the earlier models
### P values indicate that the variables season, weekday, humidity, windspeed, temperature and season
#### are positively contributing to the hiring of the bikes
### hence we reject the null hypothesis that the variables have no impact on hiring of the bikes.

#### Predict the model ########
l_predictions = predict(l_model2step,test)

#Plot a graph for actual vs predicted values
plot(test$cnt,type="l",lty=2,col="green")
lines(l_predictions,col="blue")

### graph shows that the test data predicting better than the actual data

library(DMwR)

summary(b$cnt)

regr.eval(test[,16],l_predictions,stats=c('mae',"rmse","mape"))









  















