## Michael J. Bernier
## DSCI 302 Fall 2020 B
## Final Project

## Instructions
##  1. Load the dataset pmsm_temperature_data.xlsx Preview the document into memory. 
##  2. Consider the following predictors, Ambien, Coolant, u_d, u_q, motor_speed, 
##     Torque, stator_yoke, and stator_winding.  List the categorical variable from 
##     this list and convert it to a factor.  
##  3. Calculate the minimum, maximum, mean, median, standard deviation and three 
##     quartiles (25th, 50th and 75th percentiles) of Pm. 
##  4. Calculate the minimum, maximum, mean, median, standard deviation and three 
##     quartiles (25th, 50th and 75th percentiles) of motor_speed. 
##  5. Calculate the correlation coefficient of the two variables: motor_speed and Pm. 
##     Do they have a strong relationship? 
##  6. Calculate the frequency table of stator_yoke? What's the mode of stator_yoke 
##     variable? 
##  7. Plot the histogram and density of the Pm and add the vertical line denoting 
##     the mean using ggplot2. 
##  8. Construct the scatter plot of Pm (y-axis) against motor_speed (x-axis) and add 
##     the trend line using ggplot2. 
##  9. Plot the boxplot Pm (y-axis) against stator_yoke (x-axis) and save the graph 
##     in a file, pmyoke.jpg, using ggplot2. Are there any differences in Pm with respect
##     to stator_yoke? 
## 10. Build the following multiple linear regression models: 
##     a. Preform multiple linear regression with Pm as the response and the predictors
##        are: Ambien, Coolant, motor_speed, and Torque. Write down the math formula with
##        numerical coefficients. 
##     b. Preform multiple linear regression with Pm as the response and the predictors are: 
##        Ambien, Coolant, u_d,  motor_speed, Torque, and stator_winding. Write down the math 
##        formula with numerical coefficients. 
##     c. Preform multiple linear regression with Pm as the response and the predictors are: 
##        Ambien, Coolant, u_d, u_q, motor_speed, Torque, stator_yoke, and stator_winding. 
##        Write down the math formula with numerical coefficients. 
##     d. Which model do you recommend to the management based on adjusted R squared? 
##        Justify your answer. 
## 11. Build the following KNN models: 
##     a. split the data into training dataset (85% of the original data) and test 
##        data set (15%) 
##     b. forecast stator_yoke using Pm, Ambien, and Coolant. 
##     c. forecast the stator_yoke using Pm, Ambien, Coolant, and  motor_speed 
##     d. forecast the stator_yoke using Pm, Ambien, Coolant, u_d, u_q, motor_speed, 
##        and Torque 
##     e. Which model do you recommend to the management based on accuracy of the test 
##        data set? Justify your answer 



##  1. Load the dataset pmsm_temperature_data.xlsx Preview the document into memory. 

library(readxl)
pmsm_temperature_data <- read_excel("pmsm_temperature_data-1.xlsx")

View(pmsm_temperature_data)



##  2. Consider the following predictors, Ambien, Coolant, u_d, u_q, motor_speed, 
##     Torque, stator_yoke, and stator_winding.  List the categorical variable from 
##     this list and convert it to a factor.  

## display all variables and their classifications
sapply(pmsm_temperature_data, class)

## stator_yoke is the only non-numeric variable; converting to factor
pmsm_temperature_data$stator_yoke <- as.factor(pmsm_temperature_data$stator_yoke)



##  3. Calculate the minimum, maximum, mean, median, standard deviation and three 
##     quartiles (25th, 50th and 75th percentiles) of Pm. 

min(pmsm_temperature_data$pm)
max(pmsm_temperature_data$pm)
mean(pmsm_temperature_data$pm)
median(pmsm_temperature_data$pm)
sd(pmsm_temperature_data$pm)
quantile(pmsm_temperature_data$pm)



##  4. Calculate the minimum, maximum, mean, median, standard deviation and three 
##     quartiles (25th, 50th and 75th percentiles) of motor_speed. 

min(pmsm_temperature_data$motor_speed)
max(pmsm_temperature_data$motor_speed)
mean(pmsm_temperature_data$motor_speed)
median(pmsm_temperature_data$motor_speed)
sd(pmsm_temperature_data$motor_speed)
quantile(pmsm_temperature_data$motor_speed)



##  5. Calculate the correlation coefficient of the two variables: motor_speed and Pm. 

cor(pmsm_temperature_data$motor_speed,pmsm_temperature_data$pm)

##     Do they have a strong relationship? 

## No, they do not have a strong relationship


##  6. Calculate the frequency table of stator_yoke? What's the mode of stator_yoke 
##     variable? 

## frequency table
table(pmsm_temperature_data$stator_yoke)

## find mode
names(sort(-table(pmsm_temperature_data$stator_yoke)))[1]



##  7. Plot the histogram and density of the Pm and add the vertical line denoting 
#      the mean using ggplot2. 

library(ggplot2)

ggplot(data = pmsm_temperature_data, aes(x=pm)) +
  geom_histogram(aes(y=..density..), colour="black", fill="green") +
  geom_density(alpha = 0.2, fill="red") +
  geom_vline(aes(xintercept = mean(pm)), color="blue", linetype = "dashed", size=1)



##  8. Construct the scatter plot of Pm (y-axis) against motor_speed (x-axis) and 
##     add the trend line using ggplot2. 

ggplot(data = pmsm_temperature_data, aes(x=motor_speed, y=pm))+
  geom_point() +
  geom_smooth(method="lm", se=FALSE)



##  9. Plot the boxplot Pm (y-axis) against stator_yoke (x-axis) and save the graph 
##     in a file, pmyoke.jpg, using ggplot2. Are there any differences in Pm with 
##     respect to stator_yoke? 

ggplot(data = pmsm_temperature_data, aes(x=stator_yoke, y=pm))+
  geom_boxplot()
ggsave("pmyoke.jpg")


## Yes. The values for Pm are in the negative range when the stator_yoke value is 
## "Negative" and in the positive range when the stator_yoke value is "Positive".



## 10. Build the following multiple linear regression models: 

##     a. Preform multiple linear regression with Pm as the response and the predictors
##        are: Ambien, Coolant, motor_speed, and Torque. Write down the math formula
##        with numerical coefficients. 

lm.result1 <- lm(pm ~ ambient + coolant + motor_speed + torque, data=pmsm_temperature_data)
summary(lm.result1)

## Math formula:
## pm = 0.009985 + (0.345500 * ambient) + (0.302666 * coolant) + (0.398226 * motor_speed) + 
##      (0.072800 * torque)


##     b. Preform multiple linear regression with Pm as the response and the predictors
##        are: Ambien, Coolant, u_d,  motor_speed, Torque, and stator_winding. Write down
##        the math formula with numerical coefficients. 

lm.result2 <- lm(pm ~ ambient + coolant + u_d + motor_speed + torque + stator_winding,
                 data=pmsm_temperature_data)
summary(lm.result2)

## Math formula:
## pm = 0.006438 + (0.282715 * ambient) + (-0.014670 * coolant) + (-0.265500 * u_d) + 
##      (0.048006 * motor_speed) + (-0.289061 * torque) + (0.618445 * stator_winding)


##     c. Preform multiple linear regression with Pm as the response and the predictors
##        are: Ambien, Coolant, u_d, u_q, motor_speed, Torque, stator_yoke, and 
##        stator_winding. Write down the math formula with numerical coefficients. 

lm.result3 <- lm(pm ~ ambient + coolant + u_d + u_q + motor_speed + torque + stator_winding
                 + stator_yoke, data = pmsm_temperature_data)
summary(lm.result3)

## Math formulas:
## pm = 0.160736 + (0.286867 * ambient) + (-0.067238 * coolant) + (0.247135 * u_d) + 
##      (-0.057886 * u_q) + (0.106641 * motor_speed) + (-0.264481 * torque) + 
##      (0.530634 * stator_winding)  [stator_yoke is Positive]

## pm = -0.123766 + (0.286867 * ambient) + (-0.067238 * coolant) + (0.247135 * u_d) + 
##      (-0.057886 * u_q) + (0.106641 * motor_speed) + (-0.264481 * torque) + 
##      (0.530634 * stator_winding)  [stator_yoke is Negative]

## Note: first equation has been simplified by adding together the values for the 
##       intercept and the coefficient for stator_yokePositive (-0.123766 + 0.284502 respectively)




##     d. Which model do you recommend to the management based on adjusted R squared?
##        Justify your answer. 

## I recommend model (c) because its adjusted R squared value is the highest (0.674 
## compared to 0.6674 for model (b) and 0.4486 for model (a). The higher R squared 
## value indicates model (c) provides the best fit for the data.



## 11. Build the following KNN models: 

##     a. split the data into training dataset (85% of the original data) and test
##        data set (15%) 

## load class library
library(class)

## play it safe and normalize all of the data

## create normalizing function
NormalizedData <- function(vinput) {
  result <- (vinput - min(vinput))/(max(vinput) - min(vinput))
  return(result)
}

## normalizing data
pmsm_temperature_data$pm          <- NormalizedData(pmsm_temperature_data$pm)
pmsm_temperature_data$ambient     <- NormalizedData(pmsm_temperature_data$ambient)
pmsm_temperature_data$coolant     <- NormalizedData(pmsm_temperature_data$coolant)
pmsm_temperature_data$motor_speed <- NormalizedData(pmsm_temperature_data$motor_speed)
pmsm_temperature_data$torque      <- NormalizedData(pmsm_temperature_data$torque)
pmsm_temperature_data$u_d         <- NormalizedData(pmsm_temperature_data$u_d)
pmsm_temperature_data$u_q         <- NormalizedData(pmsm_temperature_data$u_q)

## set training sample to 85% of dataset
train.sample <- floor(0.85 * nrow(pmsm_temperature_data))


##     b. forecast stator_yoke using Pm, Ambien, and Coolant. 

## set target for forecast
fcast1.target <- pmsm_temperature_data$stator_yoke

## set predictors
predictors1 <- c("pm", "ambient", "coolant")

## select predictors
fcast1.predictors <- pmsm_temperature_data[predictors1]

## generate training model
fcast1.train <- fcast1.predictors[1:train.sample, ]

## generate testing model
fcast1.test <- fcast1.predictors[-c(1:train.sample), ]

## select corresponding labels
fcast1.cl <- fcast1.target[1:train.sample]

## computer number of neighbors
fcast1.neighbors <- floor(sqrt(nrow(pmsm_temperature_data)))

## run KNN algorithm
fcast1.knn.predict <- knn(fcast1.train, fcast1.test, fcast1.cl, k = fcast1.neighbors)

## confusion matrix/contingency table
fcast1.label <- fcast1.target[-c(1:train.sample)]
table(fcast1.label, fcast1.knn.predict)


##     c. forecast the stator_yoke using Pm, Ambien, Coolant, and  motor_speed 

## set target for forecast
fcast2.target <- pmsm_temperature_data$stator_yoke

## set predictors
predictors2 <- c("pm", "ambient", "coolant", "motor_speed")

## select predictors
fcast2.predictors <- pmsm_temperature_data[predictors2]

## generate training model
fcast2.train <- fcast2.predictors[1:train.sample, ]

## generate testing model
fcast2.test <- fcast2.predictors[-c(1:train.sample), ]

## select corresponding labels
fcast2.cl <- fcast2.target[1:train.sample]

## computer number of neighbors
fcast2.neighbors <- floor(sqrt(nrow(pmsm_temperature_data)))

## run KNN algorithm
fcast2.knn.predict <- knn(fcast2.train, fcast2.test, fcast2.cl, k = fcast2.neighbors)

## confusion matrix/contingency table
fcast2.label <- fcast2.target[-c(1:train.sample)]
table(fcast2.label, fcast2.knn.predict)


##     d. forecast the stator_yoke using Pm, Ambien, Coolant, u_d, u_q, motor_speed,
##        and Torque 

## set target for forecast
fcast3.target <- pmsm_temperature_data$stator_yoke

## set predictors
predictors3 <- c("pm", "ambient", "coolant", "motor_speed", "u_d", "u_q", "torque")

## select predictors
fcast3.predictors <- pmsm_temperature_data[predictors3]

## generate training model
fcast3.train <- fcast3.predictors[1:train.sample, ]

## generate testing model
fcast3.test <- fcast3.predictors[-c(1:train.sample), ]

## select corresponding labels
fcast3.cl <- fcast3.target[1:train.sample]

## computer number of neighbors
fcast3.neighbors <- floor(sqrt(nrow(pmsm_temperature_data)))

## run KNN algorithm
fcast3.knn.predict <- knn(fcast3.train, fcast3.test, fcast3.cl, k = fcast3.neighbors)

## confusion matrix/contingency table
fcast3.label <- fcast3.target[-c(1:train.sample)]
table(fcast3.label, fcast3.knn.predict)


##     e. Which model do you recommend to the management based on accuracy of the 
##        test data set? Justify your answer 

## accuracy of model (b) = (849+528)/(849+50+73+528) = 1377/1500 = 0.918 = 91.8%
## accuracy of model (c) = (862+524)/(862+37+77+524) = 1386/1500 = 0.924 = 92.4%
## accuracy of model (d) = (858+525)/(858+41+76+525) = 1383/1500 = 0.922 = 92.2%

## Based on the accuracy of the test data sets, my recommendation would be model (c).
## While normally adding more predictors would be expected to improve accuracy, in this case
## the addition of u_d, u_q, and torque caused the accuracy of model (d) to decrease slightly,
## meaning those values were contradictory to the patterns found in the other predictors


