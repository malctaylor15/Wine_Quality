# Set working directory 
path <- "C:/Users/board/Desktop/Kaggle/Wine_Quality"
setwd(path)
set.seed(4)
library(data.table)

data <- fread("winequality-white.csv")
dim(data)

summary(data)
summary(data$quality)
hist(data$quality)
# pairs(data)
n_obs <- nrow(data)

train_rows <- sample(n_obs, 0.75*n_obs)
train <- data[train_rows, ]

fit1 <- lm(quality~ alcohol, data = train)
summary(fit1)
plot(train$alcohol, train$quality)
abline(fit1, lwd = 2, col = "red")

fit2 <- lm(quality ~. , data = train)
summary(fit2)

##################################

quality_binom <- ifelse(data$quality < 6, 0, 1)

hist(data$alcohol)
library(MASS)
alochol_beta_params <- fitdistr(data$alcohol, "beta", )


estBetaParams <- function(mu, var) {
  alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
  beta <- alpha * (1 / mu - 1)
  return(params = list(alpha = alpha, beta = beta))
}

scaled_alc <- (data$alcohol - min(data$alcohol))/(1.001*max(data$alcohol) - min(data$alcohol)) +    0.001

hist(scaled_alc, breaks = 30)

summary(scaled_alc)
mean(scaled_alc); sd(scaled_alc)

beta_prac <- estBetaParams(mean(scaled_alc), sd(scaled_alc))
beta_prac

beta_prac2 <- fitdistr(scaled_alc, "beta" , start = list(shape1 = 2, shape2 = 4))
beta_prac2

hist(scaled_alc, probability = T, breaks = 30)
x_beta <- seq(0, 1, by = 0.02)
beta_1 <- dbeta(x_beta, shape1 = beta_prac$alpha, shape2 = beta_prac$beta)
beta_2 <- dbeta(x_beta, shape1 = beta_prac2$estimate[1], shape2 = beta_prac2$estimate[1])

beta_3 <- dbeta(x_beta, shape1 = 2.5, shape2 = 4.5) 


lines(x_beta, beta_1, col = "blue", lwd = 3)
lines(x_beta, beta_2, col = "red" , lwd = 3)
lines(x_beta, beta_3, col = "black", lwd = 4)

legend("topright", c("Mean/ Var", "FitDistr"), col = c("blue", "red"), lwd = 3)

ks.test(beta_3, data$alcohol)



