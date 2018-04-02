setwd('C:/Users/redwa_000/Downloads/Max/0 UVa/Text Mining/text-mining-2') #omitted
data <- read.table("output.csv", header=TRUE, sep=",")
colnames(data)<-c("accuracy","log.likelihood")
attach(data)

plot(log(accuracy), log.likelihood, xlab="log(accuracy)", ylab="log-likelihood")
