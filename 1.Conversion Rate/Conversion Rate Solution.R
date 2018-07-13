#install.packages("dplyr")
#install.packages("rpart")
#install.packages("ggplot2")
#install.packages("randomForest")

require(dplyr)
require(rpart)
require(ggplot2)
require(randomForest)

dataset = read.csv("conversion_data.csv")
head(dataset)
str(dataset)
summary(dataset)

sort(unique(dataset$age), decreasing = TRUE)

subset(dataset, age>79)

dataset = subset(dataset, age<80)

data_country = dataset %>%
              group_by(country) %>%
              summarise(conversion_rate=mean(converted))

qplot(country, conversion_rate, data = data_country, geom = "col", fill=country)


data_age = dataset %>%
  group_by(age) %>%
  summarise(conversion_rate=mean(converted))

qplot(age, conversion_rate, data = data_age, geom = "col", fill=age)


data_pages = dataset %>%
  group_by(total_pages_visited) %>%
  summarise(conversion_rate=mean(converted))

qplot(total_pages_visited, conversion_rate, data = data_pages, geom = "line")

dataset$converted = as.factor(dataset$converted)
dataset$new_user = as.factor(dataset$new_user)
levels(dataset$country)[levels(dataset$country)=="Germany"]="DE"

train_sample = sample(nrow(dataset), size = nrow(dataset)*0.66)
train_data = dataset[train_sample,]
test_data = dataset[-train_sample,]

rf = randomForest(y=train_data$converted, x = train_data[, -ncol(train_data)],
                  ytest = test_data$converted, xtest = test_data[, -ncol(test_data)],
                  ntree = 100, mtry = 3, keep.forest = T)

rf

varImpPlot(rf, type = 2)

rf = randomForest(y=train_data$converted, x = train_data[,-c(5, ncol(train_data))],
                  ytest = test_data$converted, xtest = test_data[, -c(5, ncol(test_data))],
                  ntree = 100, mtry = 3, keep.forest = T, classwt = c(0.7, 0.3))

rf

varImpPlot(rf, type = 2)

op <- par(mfrow=c(2,2))
partialPlot(rf, train_data, country, 1)
partialPlot(rf, train_data, age, 1)
partialPlot(rf, train_data, new_user, 1)
partialPlot(rf, train_data, source, 1)

tree = rpart(dataset$converted ~ ., dataset[, -c(5, ncol(dataset))],
             control = rpart.control(maxdepth = 3),
             parms = list(prior = c(0.7, 0.3))
             )
tree