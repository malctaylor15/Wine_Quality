
# Set working directory 
path <- "C:/Users/board/Desktop/Kaggle/Wine_Quality"
setwd(path)

#install.packages("data.table")
library(data.table)

train <- fread("winequality-white.csv")
dim(train)

install.packages("h2o")
library(h2o)

localH2O <- h2o.init(nthreads = -1)

train.h2o <- as.h2o(train)
colnames(train.h2o)

y.index <- 12
x.index <- c(1:11)

regression.model <- h2o.glm(y = y.index, x = x.index, training_frame = train.h2o, family = "gaussian")

h2o.performance(regression.model)


h2o.shutdown()
y
