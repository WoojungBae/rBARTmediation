\name{prBARTmediation}
\title{Predicting new observations with a previously fitted rBARTmediation model}
\alias{prBARTmediation}
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
prBARTmediation(object0,   # object from rBARTmediation
                object1,   # object from rBARTmediation
                X.test,   # matX matrix to predict at
                Uindex)
}
\arguments{
  \item{object0}{\code{object0} returned from \code{rBARTmediation}.}
  \item{object1}{\code{object1} returned from \code{rBARTmediation}.}
  \item{X.test}{Matrix of covariates to predict \eqn{y} for.}
  \item{Uindex}{Integer indices specifying the random effects.}
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
  Returns a matrix of predictions corresponding to \code{x.test}.
}
\author{
  Woojung Bae: \email{matt.woojung@gmail.com}
}
\seealso{
  \code{\link{rBARTmediation}}
  \code{\link{predict.rBARTmediation}}
}
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}