\name{predict.rBARTmediation}
\title{Predicting new observations with a previously fitted rBARTmediation model}
\alias{predict.rBARTmediation}
\description{
BART is a Bayesian \dQuote{sum-of-trees} model.\cr
For a numeric response \eqn{y}, we have
\eqn{y = f(x) + \epsilon}{y = f(x) + e},
where \eqn{\epsilon \sim N(0,\sigma^2)}{e ~ N(0,sigma^2)}.\cr

\eqn{f} is the sum of many tree models.
The goal is to have very flexible inference for the uknown
function \eqn{f}.

In the spirit of \dQuote{ensemble models},
each tree is constrained by a prior to be a weak learner
so that it contributes a
small amount to the overall fit.
}
\usage{
   \method{predict}{rBARTmediation}(object, newdata, Uindex, ...)
}
\arguments{
   \item{object}{ \code{object} returned from previous BART fit.}
   \item{newdata}{ Matrix of covariates to predict \eqn{y} for.}
   \item{Uindex}{Integer indices specifying the random effects.}
   \item{...}{ Other arguments which will be passed on to \code{prBARTmediation}.}
}
\details{
   BART is an Bayesian MCMC method.
   At each MCMC interation, we produce a draw from the joint posterior
   \eqn{(f,\sigma) | (x,y)}{(f,sigma) \| (x,y)} in the numeric \eqn{y} case
   and just \eqn{f} in the binary \eqn{y} case.

   Thus, unlike a lot of other modelling methods in R, we do not produce a single model object
   from which fits and summaries may be extracted.  The output consists of values
   \eqn{f^*(x)}{f*(x)} (and \eqn{\sigma^*}{sigma*} in the numeric case) where * denotes a particular draw.
   The \eqn{x} is either a row from the training data (x.train) or the test data (x.test).
}
\value{
  Returns a matrix of predictions corresponding to \code{newdata}.
}
\author{
  Woojung Bae: \email{matt.woojung@gmail.com}
}
\seealso{
  \code{\link{rBARTmediation}}
  \code{\link{prBARTmediation}}
}
\examples{
  library(rBARTmediation)
  
  # simulate data (example from Friedman MARS paper)
  f = function(x){
    10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2 + 10*x[,4] + 5*x[,5]
  }
  
  g = function(x){
    10*cos(pi*x[,1]*x[,7]) + 20*(x[,8]-.5)^2 + 10*x[,9] + 5*x[,10]
  }
  
  set.seed(0)
  
  # number of observations
  n = 1000
  
  # 10 variables, only first 5 matter
  J = 10
  u0 = sample(1:J, n, replace = TRUE)
  z = rbinom(J, 1, 0.5)
  z = z[u0]
  x = matrix(runif(n*9),n,9)
  c = x[,1:6]
  v = x[,7:9]
  
  # m = f(matx) + Msigma * z where Msigma = 1.0 and z ~ N(0,1) and matx = cbind(z, x)
  m0 = f(cbind(0, x)) + 1*rnorm(n)
  m1 = f(cbind(1, x)) + 1*rnorm(n)
  m = ifelse(z==1,m1,m0)
  
  # y = g(matx) * m + Ysigma * z where Ysigma = 2.0 and z ~ N(0,1) and matx = cbind(z, x)
  y00 = g(cbind(0, x)) * m0 + 2 * rnorm(n)
  y10 = g(cbind(1, x)) * m0 + 2 * rnorm(n)
  y11 = g(cbind(1, x)) * m1 + 2 * rnorm(n)
  y = ifelse(z==1,y11,y00)
  
  # test BART with token run to ensure installation works
  BARTfit = rBARTmediation(y,m,z,c,v,u0,nskip=5,ndpost=5)
  BARTfitPRED = predict(BARTfit,x,u0)

  \dontrun{
    # run BART
    BARTfit = rBARTmediation(y,m,z,c,v,u0)
    BARTfitPRED = predict(BARTfit,x,u0)
  }
}
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}
