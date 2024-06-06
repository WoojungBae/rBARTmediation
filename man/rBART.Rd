\name{rBART}
\title{Generalized BART Mixed Model for continuous outcomes}
\alias{rBART}
\alias{mc.rBART}
\description{
BART is a Bayesian \dQuote{sum-of-trees} model.\cr
For a numeric response \eqn{y}, we have
\eqn{y = f(matX) + \epsilon}{y = f(matX) + e},
where \eqn{\epsilon \sim N(0,\sigma^2)}{e ~ N(0,sigma^2)}.\cr

\eqn{f} is the sum of many tree models.
The goal is to have very flexible inference for the uknown
function \eqn{f}.

In the spirit of \dQuote{ensemble models},
each tree is constrained by a prior to be a weak learner
so that it contributes a small amount to the overall fit.
}
\usage{
rBART(Y, matX, Uindex=NULL, 
      typeY = "continuous",
      B_u=NULL,
      sparse=FALSE, theta=0, omega=1,
      a=0.5, b=1, augment=FALSE, rho=NULL,
      xinfo=matrix(0,0,0), usequants=FALSE,
      rm.const=TRUE,
      sigest=NA, sigdf=3, sigquant=0.90,
      k=2, power=2, base=0.95,
      lambda=NA, tau.num=NA,
      offsetY = NULL,
      ntree=200L, numcut=100L,
      ndpost=1e3, nskip=1e4, keepevery=1e1,
      printevery = (ndpost*keepevery)/10,
      transposed=FALSE, hostname=FALSE,
      mc.cores = 1L, nice = 19L, seed = 99L)
}

\arguments{
  \item{matX}{ Explanatory variables for training (in sample)
    data.\cr May be a matrix or a data frame, with (as usual) rows
    corresponding to observations and columns to variables.\cr If a
    variable is a factor in a data frame, it is replaced with dummies.
    Note that \eqn{q} dummies are created if \eqn{q>2} and one dummy
    created if \eqn{q=2} where \eqn{q} is the number of levels of the
    factor.  \code{rBART} will generate draws of \eqn{f(matX)} for each
    \eqn{matX} which is a row of matX.}
  \item{Y}{
    Continuous or binary dependent variable for training (in sample) data.\cr
    If \eqn{y} is numeric, then a continuous BART model is fit (Normal errors).\cr
    If \eqn{y} is binary (has only 0's and 1's), then a binary BART model
    with a probit link is fit by default: you can over-ride the default via the
    argument \code{type} to specify a logit BART model.
  }
  \item{Uindex}{
    Integer indices specifying the random effects.
  }
  \item{typeY}{ You can use this argument to specify the type of fit:
    "continuous", "binary", "multinomial". 
  }
  \item{B_u}{
    The prior for the standard deviation of the random effects
    is \eqn{U(0, B_u)}.
  }
  \item{sparse}{Whether to perform variable selection based on a
    sparse Dirichlet prior rather than simply uniform; see Linero 2016.
  }
  \item{theta}{Set \eqn{theta} parameter; zero means random.}
  \item{omega}{Set \eqn{omega} parameter; zero means random.}
  \item{a}{Sparse parameter for \eqn{Beta(a, b)} prior:
    \eqn{0.5<=a<=1} where lower values inducing more sparsity.
  }
  \item{b}{Sparse parameter for \eqn{Beta(a, b)} prior; typically,
    \eqn{b=1}.
  }
  \item{rho}{Sparse parameter: typically \eqn{rho=p} where \eqn{p} is the
    number of covariates under consideration.
  }
  \item{augment}{Whether data augmentation is to be performed in sparse
    variable selection.
  }
  \item{xinfo}{ You can provide the cutpoints to BART or let BART
    choose them for you.  To provide them, use the \code{xinfo}
    argument to specify a list (matrix) where the items (rows) are the
    covariates and the contents of the items (columns) are the
    cutpoints.  
  }
  \item{usequants}{ If \code{usequants=FALSE}, then the
    cutpoints in \code{xinfo} are generated uniformly; otherwise,
    if \code{TRUE}, uniform quantiles are used for the cutpoints. 
  }
  \item{rm.const}{ Whether or not to remove constant variables.}
  \item{sigest}{ The prior for the error variance
    (\eqn{sigma^2}{sigma\^2}) is inverted chi-squared (the standard
    conditionally conjugate prior).  The prior is specified by choosing
    the degrees of freedom, a rough estimate of the corresponding
    standard deviation and a quantile to put this rough estimate at.  If
    \code{sigest=NA} then the rough estimate will be the usual least squares
    estimator.  Otherwise the supplied value will be used.
    Not used if \eqn{y} is binary.
  }
  \item{sigdf}{
    Degrees of freedom for error variance prior.
    Not used if \eqn{y} is binary.
  }
  \item{sigquant}{ The quantile of the prior that the rough estimate
    (see \code{sigest}) is placed at.  The closer the quantile is to 1, the more
    aggresive the fit will be as you are putting more prior weight on
    error standard deviations (\eqn{sigma}) less than the rough
    estimate.  Not used if \eqn{y} is binary.
  }
  \item{k}{ For numeric \eqn{y}, \code{k} is the number of prior
    standard deviations \eqn{E(Y|matX) = f(matX)} is away from +/-0.5.  The
    response, Y, is internally scaled to range from -0.5 to
    0.5.  For binary \eqn{y}, \code{k} is the number of prior standard
    deviations \eqn{f(matX)} is away from +/-3.  The bigger \code{k} is, the more
    conservative the fitting will be.
  }
  \item{power}{
    Power parameter for tree prior.
  }
  \item{base}{
    Base parameter for tree prior.
  }
  \item{lambda}{
    The scale of the prior for the variance.  If \code{lambda} is zero,
    then the variance is to be considered fixed and known at the given
    value of \code{sigest}.  Not used if \eqn{y} is binary.
  }
  \item{tau.num}{ The numerator in the \code{tau} definition, i.e.,
    \code{tau=tau.num/(k*sqrt(ntree))}. 
  }
  %% \item{tau.interval}{
  %%   The width of the interval to scale the variance for the terminal
  %%   leaf values.  Only used if \eqn{y} is binary.}
  \item{offsetY}{ Continous BART operates on Y centered by
    \code{offsetY} which defaults to \code{mean(Y)}.  With binary
    BART, the centering is \eqn{P(Y=1 | matX) = F(f(matX) + offsetY)} where
    \code{offsetY} defaults to \code{F^{-1}(mean(Y))}.  You can use
    the \code{offsetY} parameter to over-ride these defaults.
  }
  \item{ntree}{
    The number of trees in the sum.
  }
  \item{numcut}{ The number of possible values of \eqn{c} (see
    \code{usequants}).  If a single number if given, this is used for all
    variables.  Otherwise a vector with length equal to
    \code{ncol(matX)} is required, where the \eqn{i^{th}}{i^th}
    element gives the number of \eqn{c} used for the \eqn{i^{th}}{i^th}
    variable in matX.  If usequants is false, numcut equally
    spaced cutoffs are used covering the range of values in the
    corresponding column of matX.  If \code{usequants} is true, then
    \eqn{min(numcut, the number of unique values in the corresponding
    columns of matX - 1)} values are used.  
  }
  \item{ndpost}{
    The number of posterior draws returned.
  }
  \item{nskip}{
    Number of MCMC iterations to be treated as burn in.
  }
  \item{printevery}{
    As the MCMC runs, a message is printed every printevery draws.
  }
  \item{keepevery}{
    Every keepevery draw is kept to be returned to the user.\cr
    %% A \dQuote{draw} will consist of values of the error standard deviation (\eqn{\sigma}{sigma})
    %% and \eqn{f^*(matX)}{f*(matX)}
    %% at \eqn{matX} = rows from the train(optionally) and test data, where \eqn{f^*}{f*} denotes
    %% the current draw of \eqn{f}.
  }
  \item{transposed}{
    When running \code{rBART} in parallel, it is more memory-efficient
    to transpose matX and matX.test, if any, prior to
    calling \code{mc.rBART}.
  }
  \item{hostname}{
    When running on a cluster occasionally it is useful
    to track on which node each chain is running; to do so
    set this argument to \code{TRUE}.
  }
  \item{seed}{
    Setting the seed required for reproducible MCMC.
  }
  \item{mc.cores}{
    Number of cores to employ in parallel.
  }
  \item{nice}{
    Set the job niceness.  The default
    niceness is 19: niceness goes from 0 (highest) to 19 (lowest).
  }
  }
  \details{
    BART is a Bayesian MCMC method.
    At each MCMC interation, we produce a draw from the joint posterior
    \eqn{(f,\sigma) | (matX,y)}{(f,sigma) \| (matX,y)} in the numeric \eqn{y} case
    and just \eqn{f} in the binary \eqn{y} case.
    Thus, unlike a lot of other modelling methods in R, we do not produce
    a single model object from which fits and summaries may be extracted.
    The output consists of values \eqn{f^*(matX)}{f*(matX)} (and
    \eqn{\sigma^*}{sigma*} in the numeric case) where * denotes a
    particular draw.  The \eqn{matX} is either a row from the training data,
    matX or the test data, matX.test.
    For matX/matX.test with missing data elements, \code{rBART}
    will singly impute them with hot decking. For one or more missing
    covariates, record-level hot-decking imputation \cite{deWaPann11} is
    employed that is biased towards the null, i.e., nonmissing values
    from another record are randomly selected regardless of the
    outcome. Since \code{mc.rBART} runs multiple \code{rBART} threads in
    parallel, \code{mc.rBART} performs multiple imputation with hot
    decking, i.e., a separate imputation for each thread.  This
    record-level hot-decking imputation is biased towards the null, i.e.,
    nonmissing values from another record are randomly selected
    regardless of Y.
  }
  \value{
  %% The \code{plot} method sets mfrow to c(1,2) and makes two plots.\cr
  %% The first plot is the sequence of kept draws of \eqn{\sigma}{sigma}
  %% including the burn-in draws.  Initially these draws will decline as BART finds fit
  %% and then level off when the MCMC has burnt in.\cr
  %% The second plot has \eqn{y} on the horizontal axis and posterior intervals for
  %% the corresponding \eqn{f(matX)} on the vertical axis.
  \code{rBART} returns an object of type \code{rBART} which is
    essentially a list. % assigned class \sQuote{bart}.
    In the numeric \eqn{y} case, the list has components:
  \item{yhat.train}{
    A matrix with ndpost rows and nrow(matX) columns.
    Each row corresponds to a draw \eqn{f^*}{f*} from the posterior of \eqn{f}
    and each column corresponds to a row of matX.
    The \eqn{(i,j)} value is \eqn{f^*(matX)}{f*(matX)} for the \eqn{i^{th}}{i\^th} kept draw of \eqn{f}
    and the \eqn{j^{th}}{j\^th} row of matX.\cr
    Burn-in is dropped.
  }
  \item{yhat.test}{Same as yhat.train but now the matX's are the rows of the test data.}
  \item{yhat.train.mean}{train data fits = mean of yhat.train columns.}
  \item{yhat.test.mean}{test data fits = mean of yhat.test columns.}
  \item{sigma}{post burn in draws of sigma, length = ndpost.}
  \item{first.sigma}{burn-in draws of sigma.}
  \item{varcount}{a matrix with ndpost rows and nrow(matX) columns.
    Each row is for a draw. For each variable (corresponding to the columns),
    the total count of the number of times
    that variable is used in a tree decision rule (over all trees) is given.
  }
  \item{sigest}{
    The rough error standard deviation (\eqn{\sigma}{sigma}) used in the prior.
  }
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
  \code{\link{prBART}}
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
  u0 = sample(1:5, n, replace = TRUE)
  matX = matrix(runif(n*10),n,10) 
  Ey = f(matX)
  y = Ey + sigma*rnorm(n) + u[u0]
  
  # compare lm fit to BART later
  lmFit = lm(y~.,data.frame(matX,y)) 
  
  ##test BART with token run to ensure installation works
  bartFit = rBART(y,matX,Uindex=u0,nskip=5,ndpost=5)
  
  \dontrun{
  ##run BART
  bartFit = rBART(y,matX,Uindex=u0)
  }
}
\keyword{nonparametric}
\keyword{tree}
\keyword{regression}
\keyword{nonlinear}