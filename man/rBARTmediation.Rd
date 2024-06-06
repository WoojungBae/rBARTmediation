\name{rBARTmediation}
\title{Mediation Analysis using Generalized BART Mixed Model for continuous outcomes}
\alias{rBARTmediation}
\description{
BART is a Bayesian \dQuote{sum-of-trees} model.\cr
For a numeric response \eqn{m}, we have
\eqn{m = f(X) + \epsilon}{m = f(X) + e},
where \eqn{\epsilon \sim N(0,\sigma^2)}{e ~ N(0,sigma^2)}.\cr

\eqn{f} is the sum of many tree models.
The goal is to have very flexible inference for the uknown
function \eqn{f}.

In the spirit of \dQuote{ensemble models},
each tree is constrained by a prior to be a weak learner
so that it contributes a small amount to the overall fit.
}
\usage{
rBARTmediation(Y, M, Z, C, V, Uindex=NULL, 
               typeY = "continuous",
               typeM = "continuous",
               B_uM=NULL, B_uY=NULL,
               sparse=FALSE, theta=0, omega=1,
               a=0.5, b=1, augment=FALSE, 
               matXrho=NULL, matMrho=NULL,
               xinfo=matrix(0,0,0), usequants=FALSE,
               matXrm.const=TRUE, matMrm.const=TRUE,
               Msigest=NA, Ysigest=NA, 
               sigdf=3, sigquant=0.90,
               k=2, power=2, base=0.95,
               Mlambda=NA, Ylambda=NA,  
               Mtau.num=NA, Ytau.num=NA,
               Moffset = NULL, Yoffset = NULL,
               ntree=200L, 
               matXnumcut=100L, matMnumcut=100L,
               ndpost=1e3, nskip=1e4, keepevery=1e1,
               printevery = (ndpost*keepevery)/10,
               transposed=FALSE, hostname=FALSE,
               mc.cores = 1L, nice = 19L, seed = 99L)
}
\arguments{
  \item{Y}{Continuous or binary dependent variable for training (in sample) data.\cr
    If \eqn{y} is numeric, then a continuous BART model is fit (Normal errors).\cr
    If \eqn{y} is binary (has only 0's and 1's), then a binary BART model
    with a probit link is fit by default: you can over-ride the default via the
    argument \code{type} to specify a logit BART model.}
  \item{M}{Continuous or binary dependent variable for training (in sample) data.\cr
    If \eqn{m} is numeric, then a continuous BART model is fit (Normal errors).\cr
    If \eqn{m} is binary (has only 0's and 1's), then a binary BART model
    with a probit link is fit by default: you can over-ride the default via the
    argument \code{type} to specify a logit BART model.}
  \item{Z}{Cluster-level treatment (Binary).}
  \item{C}{Individual-level:
    Explanatory variables for training (in sample)
    data.\cr May be a matrix or a data frame, with (as usual) rows
    corresponding to observations and columns to variables.\cr If a
    variable is a factor in a data frame, it is replaced with dummies.
    Note that \eqn{q} dummies are created if \eqn{q>2} and one dummy
    created if \eqn{q=2} where \eqn{q} is the number of levels of the
    factor.  \code{rBARTmediation} will generate draws of \eqn{f(X)} for each
    \eqn{matX} which is a row of \code{matX = (Z, X)}.}
  \item{V}{Cluster-level confounders:
    Explanatory variables for training (in sample)
    data.\cr May be a matrix or a data frame, with (as usual) rows
    corresponding to observations and columns to variables.\cr If a
    variable is a factor in a data frame, it is replaced with dummies.
    Note that \eqn{q} dummies are created if \eqn{q>2} and one dummy
    created if \eqn{q=2} where \eqn{q} is the number of levels of the
    factor.  \code{rBARTmediation} will generate draws of \eqn{f(X)} for each
    \eqn{matX} which is a row of \code{matX = (Z, X)}.}
  \item{Uindex}{Integer indices specifying the random effects.}
  \item{typeY}{ You can use this argument to specify the type of fit:
    "continuous", "binary", "multinomial". 
  }
  \item{typeM}{ You can use this argument to specify the type of fit:
    "continuous", "binary", "multinomial". 
  }
  \item{B_uM}{The prior for the standard deviation of the random effects (\eqn{m})
    is \eqn{U(0, B_uM)}.}
  \item{B_uY}{The prior for the standard deviation of the random effects (\eqn{y})
    is \eqn{U(0, B_uY)}.}
  \item{sparse}{Whether to perform variable selection based on a
    sparse Dirichlet prior rather than simply uniform; see Linero 2016.}
  \item{theta}{Set \eqn{theta} parameter; zero means random.}
  \item{omega}{Set \eqn{omega} parameter; zero means random.}
  \item{a}{Sparse parameter for \eqn{Beta(a, b)} prior:
    \eqn{0.5<=a<=1} where lower values inducing more sparsity.}
  \item{b}{Sparse parameter for \eqn{Beta(a, b)} prior; typically,
    \eqn{b=1}.}
  \item{matXrho}{Sparse parameter: typically \eqn{rho=pm} where \eqn{pm} is the
    number of covariates under consideration.}
  \item{matMrho}{Sparse parameter: typically \eqn{rho=py} where \eqn{py} is the
    number of covariates under consideration.}
  \item{augment}{Whether data augmentation is to be performed in sparse
    variable selection.}
  \item{xinfo}{You can provide the cutpoints to BART or let BART
    choose them for you.  To provide them, use the \code{xinfo}
    argument to specify a list (matrix) where the items (rows) are the
    covariates and the contents of the items (columns) are the
    cutpoints.}
  \item{usequants}{If \code{usequants=FALSE}, then the
    cutpoints in \code{xinfo} are generated uniformly; otherwise,
    if \code{TRUE}, uniform quantiles are used for the cutpoints. }
  \item{matXrm.const}{Whether or not to remove constant variables.}
  \item{matMrm.const}{Whether or not to remove constant variables.}
  \item{Msigest}{The prior for the error variance
    (\eqn{sigma^2}{sigma\^2}) is inverted chi-squared (the standard
    conditionally conjugate prior). The prior is specified by choosing
    the degrees of freedom, a rough estimate of the corresponding
    standard deviation and a quantile to put this rough estimate at.If
    \code{Msigest=NA} then the rough estimate will be the usual least squares
    estimator.  Otherwise the supplied value will be used.
    Not used if \eqn{m} is binary.}
  \item{Ysigest}{The prior for the error variance
    (\eqn{sigma^2}{sigma\^2}) is inverted chi-squared (the standard
    conditionally conjugate prior). The prior is specified by choosing
    the degrees of freedom, a rough estimate of the corresponding
    standard deviation and a quantile to put this rough estimate at.If
    \code{Ysigest=NA} then the rough estimate will be the usual least squares
    estimator.  Otherwise the supplied value will be used.
    Not used if \eqn{y} is binary.}
  \item{sigdf}{Degrees of freedom for error variance prior. Not used if \eqn{m} is binary.}
  \item{sigquant}{The quantile of the prior that the rough estimate
    (see \code{sigest}) is placed at.  The closer the quantile is to 1, the more
    aggresive the fit will be as you are putting more prior weight on
    error standard deviations (\eqn{sigma}) less than the rough
    estimate.  Not used if \eqn{m} is binary. }
  \item{k}{For numeric \eqn{m}, \code{k} is the number of prior
    standard deviations \eqn{E(Y|X) = f(X)} is away from +/-0.5.  The
    response, M, is internally scaled to range from -0.5 to
    0.5.  For binary \eqn{m}, \code{k} is the number of prior standard
    deviations \eqn{f(X)} is away from +/-3.  The bigger \code{k} is, the more
    conservative the fitting will be.  }
  \item{power}{Power parameter for tree prior.}
  \item{base}{Base parameter for tree prior.}
  \item{Mlambda}{The scale of the prior for the variance.  If \code{Mlambda} is zero,
    then the variance is to be considered fixed and known at the given
    value of \code{Msigest}.  Not used if \eqn{m} is binary.}
  \item{Ylambda}{The scale of the prior for the variance.  If \code{Ylambda} is zero,
    then the variance is to be considered fixed and known at the given
    value of \code{Ysigest}.  Not used if \eqn{y} is binary.}
  \item{Mtau.num}{The numerator in the \code{Mtau} definition, i.e.,
    \code{Mtau=Mtau.num/(sqrt(ntree))}. }
  \item{Ytau.num}{The numerator in the \code{Ytau} definition, i.e.,
    \code{Ytau=Ytau.num/(sqrt(ntree))}. }
  \item{Moffset}{Continous BART operates on M centered by
    \code{Moffset} which defaults to \code{mean(M)}.  With binary
    BART, the centering is \eqn{P(Y=1 | X) = F(f(X) + offset)} where
    \code{Moffset} defaults to \code{F^{-1}(mean(M))}.  You can use
    the \code{Moffset} parameter to over-ride these defaults.}
  \item{Yoffset}{Continous BART operates on Y centered by
    \code{Yoffset} which defaults to \code{mean(Y)}.  With binary
    BART, the centering is \eqn{P(Y=1 | X) = F(f(X) + offset)} where
    \code{Yoffset} defaults to \code{F^{-1}(mean(M))}.  You can use
    the \code{Yoffset} parameter to over-ride these defaults.}
  \item{ntree}{The number of trees in the sum.}
  \item{matXnumcut}{The number of possible values of \eqn{x} (see
    \code{usequants}).  If a single number if given, this is used for all
    variables.  Otherwise a vector with length equal to
    \code{ncol(X)} is required, where the \eqn{i^{th}}{i^th}
    element gives the number of \eqn{x} used for the \eqn{i^{th}}{i^th}
    variable in X.  If usequants is false, numcut equally
    spaced cutoffs are used covering the range of values in the
    corresponding column of X.  If \code{usequants} is true, then
    \eqn{min(numcut, the number of unique values in the corresponding
    columns of X - 1)} values are used.}
  \item{matMnumcut}{The number of possible values of \eqn{x} (see
    \code{usequants}).  If a single number if given, this is used for all
    variables.  Otherwise a vector with length equal to
    \code{ncol(matM)} is required, where the \eqn{i^{th}}{i^th}
    element gives the number of \eqn{x} used for the \eqn{i^{th}}{i^th}
    variable in matM.  If usequants is false, numcut equally
    spaced cutoffs are used covering the range of values in the
    corresponding column of matM.  If \code{usequants} is true, then
    \eqn{min(numcut, the number of unique values in the corresponding
    columns of matM - 1)} values are used.}
  \item{ndpost}{The number of posterior draws returned.}
  \item{nskip}{Number of MCMC iterations to be treated as burn in.}
  \item{printevery}{As the MCMC runs, a message is printed every printevery draws.}
  \item{keepevery}{Every keepevery draw is kept to be returned to the user.}
  \item{transposed}{When running \code{rBARTmediation} in parallel, it is more memory-efficient
    to transpose X and X.test.}
  \item{hostname}{When running on a cluster occasionally it is useful
    to track on which node each chain is running; to do so
    set this argument to \code{TRUE}.}
  \item{seed}{Setting the seed required for reproducible MCMC.}
  \item{mc.cores}{Number of cores to employ in parallel.}
  \item{nice}{Set the job niceness. The default niceness is 19: 
    niceness goes from 0 (highest) to 19 (lowest).}
}
\details{
  BART is a Bayesian MCMC method.
  At each MCMC interation, we produce a draw from the joint posterior
  \eqn{(f,\sigma) | (X,m)}{(f,sigma) \| (X,m)} in the numeric \eqn{m} case
  and just \eqn{f} in the binary \eqn{m} case.
  
  Thus, unlike a lot of other modelling methods in R, we do not produce
  a single model object from which fits and summaries may be extracted.
  The output consists of values \eqn{f^*(X)}{f*(X)} (and
  \eqn{\sigma^*}{sigma*} in the numeric case) where * denotes a
  particular draw.  The \eqn{X} is either a row from the training data,
  X or the test data, X.test.
  
  For X/X.test with missing data elements, \code{rBARTmediation}
  will singly impute them with hot decking. For one or more missing
  covariates, record-level hot-decking imputation \cite{deWaPann11} is
  employed that is biased towards the null, i.e., nonmissing values
  from another record are randomly selected regardless of the
  outcome. Since \code{mc.rBARTmediation} runs multiple \code{rBARTmediation} threads in
  parallel, \code{mc.rBARTmediation} performs multiple imputation with hot
  decking, i.e., a separate imputation for each thread.  This
  record-level hot-decking imputation is biased towards the null, i.e.,
  nonmissing values from another record are randomly selected
  regardless of M.
}

\value{
  \code{rBARTmediation} returns an object of type \code{rBARTmediation} which is
  essentially a list. % assigned class \sQuote{bart}.
  In the numeric \eqn{m} case, the list has components:
  \item{yhat.train}{
    A matrix with ndpost rows and nrow(X) columns.
    Each row corresponds to a draw \eqn{f^*}{f*} from the posterior of \eqn{f}
    and each column corresponds to a row of X.
    The \eqn{(i,j)} value is \eqn{f^*(X)}{f*(X)} for the \eqn{i^{th}}{i\^th} kept draw of \eqn{f}
    and the \eqn{j^{th}}{j\^th} row of X.\cr
    Burn-in is dropped.}
  \item{yhat.test}{Same as yhat.train but now the X's are the rows of the test data.}
  \item{yhat.train.mean}{train data fits = mean of yhat.train columns.}
  \item{yhat.test.mean}{test data fits = mean of yhat.test columns.}
  \item{sigma}{post burn in draws of sigma, length = ndpost.}
  \item{first.sigma}{burn-in draws of sigma.}
  \item{varcount}{a matrix with ndpost rows and nrow(X) columns.
    Each row is for a draw. For each variable (corresponding to the columns),
    the total count of the number of times
    that variable is used in a tree decision rule (over all trees) is given.}
  \item{sigest}{The rough error standard deviation (\eqn{\sigma}{sigma}) used in the prior.}
}
\references{
  Chipman, H., George, E., and McCulloch R. (2010)
     Bayesian Additive Regression Trees.
     \emph{The Annals of Applied Statistics}, \bold{4,1}, 266-298 <doi:10.1214/09-AOAS285>.
  
  Chipman, H., George, E., and McCulloch R. (2006)
     Bayesian Ensemble Learning.
     Advances in Neural Information Processing Systems 19,
     Scholkopf, Platt and Hoffman, Eds., MIT Press, Cambridge, MA, 265-272.
  
  De Waal, T., Pannekoek, J. and Scholtus, S. (2011)
     Handbook of statistical data editing and imputation.
     John Wiley & Sons, Hoboken, NJ.
    
  Friedman, J.H. (1991)
     Multivariate adaptive regression splines.
     \emph{The Annals of Statistics}, \bold{19}, 1--67.
  
  Linero, A.R. (2018)
    Bayesian regression trees for high dimensional prediction and variable selection.
    \emph{JASA}, \bold{113(522)}, 626--636.
}
\author{
  Woojung Bae: \email{matt.woojung@gmail.com}
}
\seealso{
  \code{\link{rBART}}
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
