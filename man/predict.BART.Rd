\name{predict.BART}
\title{Predicting new observations with a previously fitted BART model}
\alias{predict.BART}
%\alias{predict.pbart}
%\alias{predict.lbart}
%\alias{predict.mbart}
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
\method{predict}{BART}(object, newdata, ...)
%\method{predict}{pbart}(object, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...)
%\method{predict}{lbart}(object, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...)
%\method{predict}{mbart}(object, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...)
}
\arguments{
   \item{object}{
     \code{object} returned from previous BART fit.
   }

   \item{newdata}{
   Matrix of covariates to predict \eqn{y} for.
   }

   \item{...}{
     Other arguments which will be passed on to \code{pBART}.
   }
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
%% \references{
%% Chipman, H., George, E., and McCulloch R. (2010)
%%    Bayesian Additive Regression Trees.
%%    \emph{The Annals of Applied Statistics}, \bold{4,1}, 266-298 <doi:10.1214/09-AOAS285>.

%% Chipman, H., George, E., and McCulloch R. (2006)
%%    Bayesian Ensemble Learning.
%%    Advances in Neural Information Processing Systems 19,
%%    Scholkopf, Platt and Hoffman, Eds., MIT Press, Cambridge, MA, 265-272.

%% Friedman, J.H. (1991)
%%    Multivariate adaptive regression splines.
%%         \emph{The Annals of Statistics}, \bold{19}, 1--67.
%% }
%% \author{
%% Robert McCulloch: \email{robert.e.mcculloch@gmail.com},\cr
%% Rodney Sparapani: \email{rsparapa@mcw.edu}.
%% }
\seealso{
  \code{\link{BART}}
  \code{\link{pBART}}
  \code{\link{mc.cores.openmp}}
}
\examples{
  library(rBARTmediation)
  
  # simulate data (example from Friedman MARS paper)
  f = function(matX){
    10*sin(pi*matX[,1]*matX[,2]) + 20*(matX[,3]-.5)^2+10*matX[,4]+5*matX[,5]
  }
  
  set.seed(0)
  # number of observations
  n = 1000      
  
  # y = f(matX) + sigma * z + u , z ~ N(0,1)
  sigma = 1.0  
  # 10 variables, only first 5 matter
  J = 5
  u = rnorm(J,0,10)
  u0 = sample(1:J, n, replace = TRUE)
  matX = matrix(runif(n*10),n,10) 
  Ey = f(matX)
  y = Ey + sigma*rnorm(n) + u[u0]
  
  # compare lm fit to BART later
  lmFit = lm(y~.,data.frame(y,matX)) 
  
  ##test BART with token run to ensure installation works
  bartFit = BART(y,matX,nskip=5,ndpost=5)
  
  \dontrun{
  ##run BART
  bartFit = BART(y,matX,Uindex=u0)
  }
}
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}