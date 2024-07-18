
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

prBARTmediation = function(object,  # object from rBARTmediation
                           X.test,  # matrix X to predict at
                           Uindex){
  # object = BARTfit
  # X.test = cbind(C, V)
  # Uindex = V0
  # --------------------------------------------------
  mc.cores = 1
  
  z0 = 0
  z1 = 1
  
  N = nrow(X.test)
  J = ncol(object$uMdraw)
  n_MCMC = nrow(object$uMdraw)
  
  matXz0.test <- t(bartModelMatrix(cbind(z0,X.test)))
  matXz1.test <- t(bartModelMatrix(cbind(z1,X.test)))
  
  pm <- length(object$matXtreedraws$cutpoints)
  if(pm!=nrow(matXz0.test)) {
    stop(paste0('The number of columns in matX.test must be equal to ', pm))
  }
  
  # # --------------------------------------------------
  # # --------------------------------------------------
  # # --------------------------------------------------
  # mu.uM = as.numeric(object$mu.uM)
  # mu.uY = as.numeric(object$mu.uY)
  # sd.uM = as.numeric(object$sd.uM)
  # sd.uY = as.numeric(object$sd.uY)
  # cor.uYM = as.numeric(object$cor.uYM)
  # sig.uMM = sd.uM^{2}
  # sig.uYY = sd.uY^{2}
  # sig.uMY = cor.uYM * sd.uM * sd.uY
  # MU.uMY = cbind(mu.uM , mu.uY)
  # SIG.uMY = lapply(1:n_MCMC, function(d)
  #   rbind(c(sig.uMM[d], sig.uMY[d]), c(sig.uMY[d], sig.uYY[d])))
  # uMYreff = sapply(1:n_MCMC, function(d) t(.Call("crmvnorm", J, MU.uMY[d,], SIG.uMY[[d]])), simplify = "array")
  
  # --------------------------------------------------
  # --------------------------------------------------
  # --------------------------------------------------
  object$matXtreedraws$trees = gsub(",", " ", object$matXtreedraws$trees)
  M0res = .Call("cprBART", object$matXtreedraws, matXz0.test, mc.cores)$yhat.test + object$Moffset
  M1res = .Call("cprBART", object$matXtreedraws, matXz1.test, mc.cores)$yhat.test + object$Moffset
  Msigest = object$iMsigest
  uMreff = object$uMdraw
  for (j in 1:J) {
    whichUindex = which(Uindex==j)
    if(length(whichUindex)>0){
      uMreff_tmp = uMreff[,j] # uMYreff[1,j,] # rnorm(n_MCMC, mu.uM, sd.uM) # mu.uM # sd.uM
      M0res[,whichUindex] = M0res[,whichUindex] + uMreff_tmp
      M1res[,whichUindex] = M1res[,whichUindex] + uMreff_tmp
    }
  }
  if(object$typeM == "continuous"){
    M0.test = sapply(1:N, function(i) rnorm(n_MCMC, M0res[,i], Msigest))
    M1.test = sapply(1:N, function(i) rnorm(n_MCMC, M1res[,i], Msigest))
  } else if(object$typeM == "binary"){
    M0.test = pnorm(M0res)
    M1.test = pnorm(M1res)
    M0.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M0.test[,i]))
    M1.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M1.test[,i]))
  } else if(object$typeM == "multinomial"){
    #
  }
  
  # --------------------------------------------------
  tmp = object
  treetmp1 = tmp$matMtreedraws$trees
  treetmp2 = gsub(",", " ", treetmp1)
  treetmp3 = strsplit(treetmp1, ",")[[1]]
  treetmp4 = paste("1",treetmp3[2],treetmp3[3],treetmp3[4])
  Ysigest = object$iYsigest
  uYreff = object$uYdraw
  
  Yz0m0.test = matrix(nrow=n_MCMC,ncol=N)
  Yz1m0.test = matrix(nrow=n_MCMC,ncol=N)
  Yz1m1.test = matrix(nrow=n_MCMC,ncol=N)
  for (d in 1:n_MCMC) {
    index = 4 + d
    treetmp5 = paste(treetmp4,treetmp3[index])
    tmp$matMtreedraws$trees = treetmp5
    
    matM0z0.test = rbind(M0.test[d,], matXz0.test)
    matM0z1.test = rbind(M0.test[d,], matXz1.test)
    matM1z1.test = rbind(M1.test[d,], matXz1.test)
    
    Yz0m0res = c(.Call("cprBART", tmp$matMtreedraws, matM0z0.test, mc.cores)$yhat.test) + object$Yoffset
    Yz1m0res = c(.Call("cprBART", tmp$matMtreedraws, matM0z1.test, mc.cores)$yhat.test) + object$Yoffset
    Yz1m1res = c(.Call("cprBART", tmp$matMtreedraws, matM1z1.test, mc.cores)$yhat.test) + object$Yoffset
    for (j in 1:J) {
      whichUindex = which(Uindex==j)
      if(length(whichUindex)>0){
        uYreff_tmp = uYreff[d,j] # uMYreff[2,j,d] # rnorm(1, mu.uY[d], sd.uY[d]) # mu.uY[d] # sd.uY[d]
        Yz0m0res[whichUindex] = Yz0m0res[whichUindex] + uYreff_tmp
        Yz1m0res[whichUindex] = Yz1m0res[whichUindex] + uYreff_tmp
        Yz1m1res[whichUindex] = Yz1m1res[whichUindex] + uYreff_tmp
      }
    }
    if(object$typeY == "continuous"){
      # Yz0m0.test[d,] = Yz0m0res
      # Yz1m0.test[d,] = Yz1m0res
      # Yz1m1.test[d,] = Yz1m1res
      Yz0m0.test[d,] = rnorm(N, Yz0m0res, Ysigest[d])
      Yz1m0.test[d,] = rnorm(N, Yz1m0res, Ysigest[d])
      Yz1m1.test[d,] = rnorm(N, Yz1m1res, Ysigest[d])
    } else if(object$typeY == "binary"){
      # Yz0m0res = pnorm(Yz0m0res)
      # Yz1m0res = pnorm(Yz1m0res)
      # Yz1m1res = pnorm(Yz1m1res)
      Yz0m0.test[d,] = rbinom(N, 1, pnorm(Yz0m0res))
      Yz1m0.test[d,] = rbinom(N, 1, pnorm(Yz1m0res))
      Yz1m1.test[d,] = rbinom(N, 1, pnorm(Yz1m1res))
    } else if(object$typeY == "multinomial"){
      #
    }
  }
  
  return(list(Yz0m0.test=Yz0m0.test,
              Yz1m0.test=Yz1m0.test,
              Yz1m1.test=Yz1m1.test))
}