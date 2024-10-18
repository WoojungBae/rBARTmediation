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

prBARTmediation = function(object0,  # object from rBARTmediation
                           object1,  # object from rBARTmediation
                           X.test,  # matrix X to predict at
                           Uindex){
  # object0 = BARTfit0
  # object1 = BARTfit1
  # X.test = cbind(C, V)
  # Uindex = Uindex

  # --------------------------------------------------
  mc.cores = 1

  z0 = 0
  z1 = 1

  N = nrow(X.test)
  n_MCMC = nrow(object0$uYdraw)

  J0 = ncol(object0$uYdraw)
  J1 = ncol(object1$uYdraw)
  J = J0 + J1

  matX.test <- t(bartModelMatrix(X.test, rm.const=FALSE))

  pm <- length(object0$matXtreedraws$cutpoints)
  if(pm!=nrow(matX.test)) {
    stop(paste0('The number of columns in matX.test must be equal to ', pm))
  }

  # --------------------------------------------------
  mu.uM0 = unlist(object0$mu.uM)
  mu.uY0 = unlist(object0$mu.uY)
  sig.uM0 = unlist(object0$sig.uM)
  sig.uY0 = unlist(object0$sig.uY)
  rho.uMY0 = unlist(object0$rho.uMY)
  sig.uMM0 = sig.uM0^{2}
  sig.uYY0 = sig.uY0^{2}
  sig.uMY0 = rho.uMY0 * sig.uM0 * sig.uY0
  MU.uMY0 = cbind(mu.uM0 , mu.uY0)
  SIG.uMY0 = lapply(1:n_MCMC, function(d)
    rbind(c(sig.uMM0[d], sig.uMY0[d]), c(sig.uMY0[d], sig.uYY0[d])))
  uMYreff0 = sapply(1:n_MCMC, function(d)
    t(.Call("crmvnorm", J, c(MU.uMY0[d,]), as.matrix(SIG.uMY0[[d]]))),
    simplify = "array")

  mu.uM1 = unlist(object1$mu.uM)
  mu.uY1 = unlist(object1$mu.uY)
  sig.uM1 = unlist(object1$sig.uM)
  sig.uY1 = unlist(object1$sig.uY)
  rho.uMY1 = unlist(object1$rho.uMY)
  sig.uMM1 = sig.uM1^{2}
  sig.uYY1 = sig.uY1^{2}
  sig.uMY1 = rho.uMY1 * sig.uM1 * sig.uY1
  MU.uMY1 = cbind(mu.uM1 , mu.uY1)
  SIG.uMY1 = lapply(1:n_MCMC, function(d)
    rbind(c(sig.uMM1[d], sig.uMY1[d]), c(sig.uMY1[d], sig.uYY1[d])))
  uMYreff1 = sapply(1:n_MCMC, function(d)
    t(.Call("crmvnorm", J, c(MU.uMY1[d,]), as.matrix(SIG.uMY1[[d]]))),
    simplify = "array")

  # --------------------------------------------------
  object0$matXtreedraws$trees = gsub(",", " ", object0$matXtreedraws$trees)
  object1$matXtreedraws$trees = gsub(",", " ", object1$matXtreedraws$trees)
  M0res = .Call("cprBART", object0$matXtreedraws, matX.test, mc.cores)$yhat.test + object0$Moffset # + object$mu.uM
  M1res = .Call("cprBART", object1$matXtreedraws, matX.test, mc.cores)$yhat.test + object1$Moffset # + object$mu.uM
  Msigest0 = object0$iMsigest # sqrt(object$iMsigest^{2} + sig.uMM)
  Msigest1 = object1$iMsigest # sqrt(object$iMsigest^{2} + sig.uMM)
  # uMreff0 = object0$uMdraw
  # uMreff1 = object1$uMdraw

  for (j in 1:J) {
    whichUindex = which(Uindex==j)
    if(length(whichUindex)>0){
      uMreff_tmp = uMYreff0[1,j,] # uMYreff[1,j,] # uMreff[,j] # rnorm(n_MCMC, mu.uM, sig.uM) # mu.uM # sig.uM
      M0res[,whichUindex] = M0res[,whichUindex] + uMreff_tmp
      uMreff_tmp = uMYreff1[1,j,] # uMYreff[1,j,] # uMreff[,j] # rnorm(n_MCMC, mu.uM, sig.uM) # mu.uM # sig.uM
      M1res[,whichUindex] = M1res[,whichUindex] + uMreff_tmp
    }
  }
  if(object0$typeM == "continuous"){
    M0.test = sapply(1:N, function(i) rnorm(n_MCMC, M0res[,i], Msigest0))
    M1.test = sapply(1:N, function(i) rnorm(n_MCMC, M1res[,i], Msigest1))
  } else if(object0$typeM == "binary"){
    M0.test = pnorm(M0res)
    M1.test = pnorm(M1res)
    M0.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M0.test[,i]))
    M1.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M1.test[,i]))
  } else if(object0$typeM == "multinomial"){
    #
  }

  # --------------------------------------------------
  tmp0 = object0
  treetmp01 = tmp0$matMtreedraws$trees
  treetmp02 = gsub(",", " ", treetmp01)
  treetmp03 = strsplit(treetmp01, ",")[[1]]
  treetmp04 = paste("1",treetmp03[2],treetmp03[3],treetmp03[4])
  Ysigest0 = object0$iYsigest # sqrt(object$iYsigest^{2} + sig.uYY)
  # uYreff0 = object0$uYdraw

  tmp1 = object1
  treetmp11 = tmp1$matMtreedraws$trees
  treetmp12 = gsub(",", " ", treetmp11)
  treetmp13 = strsplit(treetmp11, ",")[[1]]
  treetmp14 = paste("1",treetmp13[2],treetmp13[3],treetmp13[4])
  Ysigest1 = object1$iYsigest # sqrt(object$iYsigest^{2} + sig.uYY)
  # uYreff1 = object1$uYdraw

  Yz0m0.test = matrix(nrow=n_MCMC,ncol=N)
  Yz1m0.test = matrix(nrow=n_MCMC,ncol=N)
  Yz1m1.test = matrix(nrow=n_MCMC,ncol=N)
  for (d in 1:n_MCMC) {
    index = 4 + d
    treetmp05 = paste(treetmp04,treetmp03[index])
    tmp0$matMtreedraws$trees = treetmp05
    treetmp15 = paste(treetmp14,treetmp13[index])
    tmp1$matMtreedraws$trees = treetmp15

    matM0.test = rbind(M0.test[d,], matX.test)
    matM1.test = rbind(M1.test[d,], matX.test)

    Yz0m0res = c(.Call("cprBART", tmp0$matMtreedraws, matM0.test, mc.cores)$yhat.test) + object0$Yoffset # + object$mu.uY
    Yz1m0res = c(.Call("cprBART", tmp1$matMtreedraws, matM0.test, mc.cores)$yhat.test) + object1$Yoffset # + object$mu.uY
    Yz1m1res = c(.Call("cprBART", tmp1$matMtreedraws, matM1.test, mc.cores)$yhat.test) + object1$Yoffset # + object$mu.uY

    for (j in 1:J) {
      whichUindex = which(Uindex==j)
      if(length(whichUindex)>0){
        uYreff_tmp = uMYreff0[2,j,d] # uMYreff[2,j,d] # uYreff[d,j] # rnorm(1, mu.uY[d], sig.uY[d]) # mu.uY[d] # sig.uY[d]
        Yz0m0res[whichUindex] = Yz0m0res[whichUindex] + uYreff_tmp
        uYreff_tmp = uMYreff1[2,j,d] # uMYreff[2,j,d] # uYreff[d,j] # rnorm(1, mu.uY[d], sig.uY[d]) # mu.uY[d] # sig.uY[d]
        Yz1m0res[whichUindex] = Yz1m0res[whichUindex] + uYreff_tmp
        Yz1m1res[whichUindex] = Yz1m1res[whichUindex] + uYreff_tmp
      }
    }

    if(object0$typeY == "continuous"){
      # Yz0m0.test[d,] = Yz0m0res
      # Yz1m0.test[d,] = Yz1m0res
      # Yz1m1.test[d,] = Yz1m1res
      Yz0m0.test[d,] = rnorm(N, Yz0m0res, Ysigest0[d])
      Yz1m0.test[d,] = rnorm(N, Yz1m0res, Ysigest1[d])
      Yz1m1.test[d,] = rnorm(N, Yz1m1res, Ysigest1[d])
    } else if(object0$typeY == "binary"){
      # Yz0m0res = pnorm(Yz0m0res)
      # Yz1m0res = pnorm(Yz1m0res)
      # Yz1m1res = pnorm(Yz1m1res)
      Yz0m0.test[d,] = rbinom(N, 1, pnorm(Yz0m0res))
      Yz1m0.test[d,] = rbinom(N, 1, pnorm(Yz1m0res))
      Yz1m1.test[d,] = rbinom(N, 1, pnorm(Yz1m1res))
    } else if(object0$typeY == "multinomial"){
      #
    }
  }

  return(list(Yz0m0.test=Yz0m0.test,
              Yz1m0.test=Yz1m0.test,
              Yz1m1.test=Yz1m1.test))
}

# ## BART: Bayesian Additive Regression Trees
# ## Copyright (C) 2017-2018 Robert McCulloch and Rodney Sparapani
# 
# ## This program is free software; you can redistribute it and/or modify
# ## it under the terms of the GNU General Public License as published by
# ## the Free Software Foundation; either version 2 of the License, or
# ## (at your option) any later version.
# 
# ## This program is distributed in the hope that it will be useful,
# ## but WITHOUT ANY WARRANTY; without even the implied warranty of
# ## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# ## GNU General Public License for more details.
# 
# ## You should have received a copy of the GNU General Public License
# ## along with this program; if not, a copy is available at
# ## https://www.R-project.org/Licenses/GPL-2
# 
# prBARTmediation = function(object,  # object from rBARTmediation
#                            X.test,  # matrix X to predict at
#                            Uindex){
#   # object = BARTfit
#   # X.test = cbind(C, V)
#   # Uindex = V0
#   # --------------------------------------------------
#   mc.cores = 1
#   
#   z0 = 0
#   z1 = 1
#   
#   N = nrow(X.test)
#   J = ncol(object$uMdraw)
#   n_MCMC = nrow(object$uMdraw)
#   
#   matXz0.test <- t(bartModelMatrix(cbind(z0,X.test)))
#   matXz1.test <- t(bartModelMatrix(cbind(z1,X.test)))
#   
#   pm <- length(object$matXtreedraws$cutpoints)
#   if(pm!=nrow(matXz0.test)) {
#     stop(paste0('The number of columns in matX.test must be equal to ', pm))
#   }
#   
#   # --------------------------------------------------
#   mu.uM = unlist(object$mu.uM)
#   mu.uY = unlist(object$mu.uY)
#   sig.uM = unlist(object$sig.uM)
#   sig.uY = unlist(object$sig.uY)
#   rho.uMY = unlist(object$rho.uMY)
#   sig.uMM = sig.uM^{2}
#   sig.uYY = sig.uY^{2}
#   sig.uMY = rho.uMY * sig.uM * sig.uY
#   MU.uMY = cbind(mu.uM , mu.uY)
#   SIG.uMY = lapply(1:n_MCMC, function(d)
#     rbind(c(sig.uMM[d], sig.uMY[d]), c(sig.uMY[d], sig.uYY[d])))
#   uMYreff = sapply(1:n_MCMC, function(d) 
#     t(.Call("crmvnorm", J, c(MU.uMY[d,]), as.matrix(SIG.uMY[[d]]))), 
#     simplify = "array")
#   
#   # --------------------------------------------------
#   object$matXtreedraws$trees = gsub(",", " ", object$matXtreedraws$trees)
#   M0res = .Call("cprBART", object$matXtreedraws, matXz0.test, mc.cores)$yhat.test + object$Moffset # + object$mu.uM
#   M1res = .Call("cprBART", object$matXtreedraws, matXz1.test, mc.cores)$yhat.test + object$Moffset # + object$mu.uM
#   Msigest = object$iMsigest # sqrt(object$iMsigest^{2} + sig.uMM)
#   uMreff = object$uMdraw
#   for (j in 1:J) {
#     whichUindex = which(Uindex==j)
#     if(length(whichUindex)>0){
#       uMreff_tmp = uMYreff[1,j,] # uMreff[,j] # rnorm(n_MCMC, mu.uM, sig.uM) # mu.uM # sig.uM
#       M0res[,whichUindex] = M0res[,whichUindex] + uMreff_tmp
#       M1res[,whichUindex] = M1res[,whichUindex] + uMreff_tmp
#     }
#   }
#   if(object$typeM == "continuous"){
#     M0.test = sapply(1:N, function(i) rnorm(n_MCMC, M0res[,i], Msigest))
#     M1.test = sapply(1:N, function(i) rnorm(n_MCMC, M1res[,i], Msigest))
#   } else if(object$typeM == "binary"){
#     M0.test = pnorm(M0res)
#     M1.test = pnorm(M1res)
#     M0.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M0.test[,i]))
#     M1.test = sapply(1:N, function(i) rbinom(n_MCMC, 1, M1.test[,i]))
#   } else if(object$typeM == "multinomial"){
#     #
#   }
#   
#   # --------------------------------------------------
#   tmp = object
#   treetmp1 = tmp$matMtreedraws$trees
#   treetmp2 = gsub(",", " ", treetmp1)
#   treetmp3 = strsplit(treetmp1, ",")[[1]]
#   treetmp4 = paste("1",treetmp3[2],treetmp3[3],treetmp3[4])
#   Ysigest = object$iYsigest # sqrt(object$iYsigest^{2} + sig.uYY)
#   uYreff = object$uYdraw
#   
#   Yz0m0.test = matrix(nrow=n_MCMC,ncol=N)
#   Yz1m0.test = matrix(nrow=n_MCMC,ncol=N)
#   Yz1m1.test = matrix(nrow=n_MCMC,ncol=N)
#   for (d in 1:n_MCMC) {
#     index = 4 + d
#     treetmp5 = paste(treetmp4,treetmp3[index])
#     tmp$matMtreedraws$trees = treetmp5
#     
#     matM0z0.test = rbind(M0.test[d,], matXz0.test)
#     matM0z1.test = rbind(M0.test[d,], matXz1.test)
#     matM1z1.test = rbind(M1.test[d,], matXz1.test)
#     
#     Yz0m0res = c(.Call("cprBART", tmp$matMtreedraws, matM0z0.test, mc.cores)$yhat.test) + object$Yoffset # + object$mu.uY
#     Yz1m0res = c(.Call("cprBART", tmp$matMtreedraws, matM0z1.test, mc.cores)$yhat.test) + object$Yoffset # + object$mu.uY
#     Yz1m1res = c(.Call("cprBART", tmp$matMtreedraws, matM1z1.test, mc.cores)$yhat.test) + object$Yoffset # + object$mu.uY
#     for (j in 1:J) {
#       whichUindex = which(Uindex==j)
#       if(length(whichUindex)>0){
#         uYreff_tmp = uMYreff[2,j,d] # uYreff[d,j] # rnorm(1, mu.uY[d], sig.uY[d]) # mu.uY[d] # sig.uY[d]
#         Yz0m0res[whichUindex] = Yz0m0res[whichUindex] + uYreff_tmp
#         Yz1m0res[whichUindex] = Yz1m0res[whichUindex] + uYreff_tmp
#         Yz1m1res[whichUindex] = Yz1m1res[whichUindex] + uYreff_tmp
#       }
#     }
#     if(object$typeY == "continuous"){
#       # Yz0m0.test[d,] = Yz0m0res
#       # Yz1m0.test[d,] = Yz1m0res
#       # Yz1m1.test[d,] = Yz1m1res
#       Yz0m0.test[d,] = rnorm(N, Yz0m0res, Ysigest[d])
#       Yz1m0.test[d,] = rnorm(N, Yz1m0res, Ysigest[d])
#       Yz1m1.test[d,] = rnorm(N, Yz1m1res, Ysigest[d])
#     } else if(object$typeY == "binary"){
#       # Yz0m0res = pnorm(Yz0m0res)
#       # Yz1m0res = pnorm(Yz1m0res)
#       # Yz1m1res = pnorm(Yz1m1res)
#       Yz0m0.test[d,] = rbinom(N, 1, pnorm(Yz0m0res))
#       Yz1m0.test[d,] = rbinom(N, 1, pnorm(Yz1m0res))
#       Yz1m1.test[d,] = rbinom(N, 1, pnorm(Yz1m1res))
#     } else if(object$typeY == "multinomial"){
#       #
#     }
#   }
#   
#   return(list(Yz0m0.test=Yz0m0.test,
#               Yz1m0.test=Yz1m0.test,
#               Yz1m1.test=Yz1m1.test))
# }