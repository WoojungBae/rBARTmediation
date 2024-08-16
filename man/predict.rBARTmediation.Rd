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
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}
