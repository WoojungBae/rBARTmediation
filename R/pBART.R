## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017-2018 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017-2018 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

pBART = function(object,  # object from rBARTmediation
                 X.test){
  # object = BARTfitM
  # X.test = matX
  # --------------------------------------------------
  mc.cores = 1
  
  N = nrow(X.test)
  n_MCMC = nrow(object$y.draw)
  
  X.test <- t(bartModelMatrix(X.test))
  
  p <- length(object$treedraws$cutpoints)
  if(p != nrow(X.test)) {
    stop(paste0('The number of columns in X.test must be equal to ', p))
  }
  
  # --------------------------------------------------
  # object$matXtreedraws$trees = gsub(",", " ", object$treedraws$trees)
  Yres = .Call("cprBART", object$treedraws, X.test, mc.cores)$yhat.test + object$offset
  Ysigest = object$sigest
  if(object$type == "continuous"){
    Y.test = sapply(1:N, function(i) rnorm(n_MCMC, Yres[,i], Ysigest))
  } else if(object$type == "binary"){
    Y.test = pnorm(Yres)
    Y.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, Y.test[,i]))
  } else if(object$type == "multinomial"){
    #
  }
  
  # return(Y.test)
  return(Yres)
}
