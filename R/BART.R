## BART: Bayesian Additive Regression Trees
## Copyright (C) 2019 Rodney Sparapani

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

BART <- function(Y, matX, 
                 typeY = "continuous", 
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
                 mc.cores = 1L, nice = 19L, seed = 99L){
  #--------------------------------------------------
  ntypeY <- as.integer(factor(typeY, levels = c("continuous", "binary", "multinomial")))
  
  if(is.na(ntypeY)){
    stop("typeY argument must be set to either 'continuous', 'binary' or 'multinomial'")
  } else if(typeY == "continuous"){
    
  } else if(typeY == "binary"){
    checkY <- unique(sort(Y))
    if(length(checkY)==2) {
      if(!all(checkY==0:1)) {
        stop('Binary Y must be coded as 0 and 1')
      }
    }
  } else if(typeY == "multinomial"){
    catsY <- unique(sort(Y))
    kY <- length(catsY)
    if(kY<2) {
      stop("there must be at least 2 categories")
    }
    # lY <- length(offsetY)
    # if(!(lY %in% c(0, kY))) {
    #   stop(paste0("length of offsetY argument must be 0 or ", kY))
    # }
  }
  
  if(typeY == "continuous"){
    offsetY <- mean(Y)
  } else if(typeY == "binary"){
    offsetY <- qnorm(offsetY)
    # offsetY <- qlogis(offsetY)
  } else if(typeY == "multinomial"){
    offsetY <- NULL
  }
  
  #--------------------------------------------------
  # data
  n <- length(Y)
  
  if(!transposed) {
    temp <- bartModelMatrix(matX, numcut, usequants=usequants,
                            xinfo=xinfo, rm.const=rm.const)
    matX <- t(temp$X)
    numcut <- temp$numcut
    xinfo <- temp$xinfo
    
    rm.const <- temp$rm.const
    rm(temp)
  } else {
    rm.const <- NULL
  }
  
  if(n!=ncol(matX)){
    stop('The length of Y and the number of rows in matX must be identical')
  }
  
  p <- nrow(matX)
  if(length(rho)==0) rho <- p
  if(length(rm.const)==0) rm.const <- 1:p
  
  u <- double(J)
  u[1] <- NaN
  #--------------------------------------------------
  # prior
  nu <- sigdf
  if(typeY == "continuous"){
    if(is.na(lambda)) {
      if(is.na(sigest)) {
        if(p < n) {
          temp <- lm(Y~., data.frame(t(matX),Y))
          sigest <- summary(temp)$sigma
        } else {
          sigest <- sd(Y)
        }
      }
      qchi <- qchisq(1.0-sigquant,nu)
      lambda <- (sigest*sigest*qchi)/nu # lambda parameter for sigma prior
    } else {
      sigest <- sqrt(lambda)
    }
    if(is.na(tau.num)) {
      tau <- (max(Y)-min(Y))/(2*k*sqrt(ntree))
    } else {
      tau <- tau.num/sqrt(ntree)
    }
  } else {
    lambda <- 1
    sigest <- 1
    tau.num <- 3 # default value of BART package ("pbart")
    tau <- tau.num/(k*sqrt(ntree))
  }
  
  #--------------------------------------------------
  if(.Platform$OS.type!='unix') {
    hostname <- FALSE
  } else if(hostname) {
    hostname <- system('hostname', intern=TRUE)
  }
  #--------------------------------------------------
  ptm <- proc.time()
  if(typeY == "continuous"){
    #--------------------------------------------------
    res <- .Call("cBART",
                 ntypeY,    # 1:"continuous"; 2:"binary"; 3:"multinomial"
                 n,         # number of observations in training data
                 p,         # dimension of x
                 matX,      # p x n training data matX
                 Y,         # 1 x n training data Y
                 ntree,
                 numcut,
                 ndpost*keepevery,
                 nskip,
                 keepevery,
                 power,
                 base,
                 offsetY,
                 tau,
                 nu,
                 lambda,
                 sigest,
                 sparse,
                 theta,
                 omega,
                 a,
                 b,
                 rho,
                 augment,
                 printevery,
                 xinfo)
    
    res$yhat.mean <- apply(res$y.draw, 2, mean)
    names(res$treedraws$cutpoints) <- dimnames(matX)[[1]]
    dimnames(res$varcount)[[2]] <- as.list(dimnames(matX)[[1]])
    dimnames(res$varprob)[[2]] <- as.list(dimnames(matX)[[1]])
    res$varcount.mean <- apply(res$varcount, 2, mean)
    res$varprob.mean <- apply(res$varprob, 2, mean)
    
  } else if(typeY == "binary"){
    #--------------------------------------------------
    res <- .Call("cBART",
                 ntypeY,    # 1:"continuous"; 2:"binary"; 3:"multinomial"
                 n,         # number of observations in training data
                 p,         # dimension of x
                 matX,      # p x n training data matX
                 Y,         # 1 x n training data Y
                 ntree,
                 numcut,
                 ndpost*keepevery,
                 nskip,
                 keepevery,
                 power,
                 base,
                 offsetY,
                 tau,
                 nu,
                 lambda,
                 sigest,
                 sparse,
                 theta,
                 omega,
                 a,
                 b,
                 rho,
                 augment,
                 printevery,
                 xinfo)
    
    res$prob <- pnorm(res$y.draw)
    # res$prob <- plogis(res$y.draw)
    res$prob.mean <- apply(res$prob, 2, mean)
    
    names(res$treedraws$cutpoints) <- dimnames(matX)[[1]]
    dimnames(res$varcount)[[2]] <- as.list(dimnames(matX)[[1]])
    dimnames(res$varprob)[[2]] <- as.list(dimnames(matX)[[1]])
    res$varcount.mean <- apply(res$varcount, 2, mean)
    res$varprob.mean <- apply(res$varprob, 2, mean)
  } else if(typeY == "multinomial"){
    #--------------------------------------------------
    res <- list()
    res$kY <- kY
    res$catsY <- catsY
    nY <- length(Y)
    pY <- nrow(matX) ## transposed

    lY <- kY - 1
    offsetY <- numeric(lY)
    res$varcount <- as.list(1:lY)
    res$varprob  <- as.list(1:lY)
    # res$varcount <- array(dim=c(ndpost, pY, lY))
    res$varcount.mean <- matrix(nrow=lY, ncol=pY)
    # res$varprob <- array(dim=c(ndpost, pY, lY))
    res$varprob.mean <- matrix(nrow=lY, ncol=pY)
    res$offsetY <- offsetY
    res$treedraws <- list()
    res$treedraws$trees <- as.list(1:lY)
    ##res$rm.const <- as.list(1:lY)
    res.list <- as.list(1:lY)

    for(k in 1:kY) {
      condY <- which(Y>=catsY[k])

      tmp.n = length(condY)
      tmp.Y = (Y[condY]==k)*1
      tmp.matX = matX[, condY]
      tmp.offsetY = offsetY[k]

      if(k<kY) {
        res.list[[k]] <-
          .Call("cBART",
                ntypeY = ntypeY,      # 1:"continuous"; 2:"binary"; 3:"multinomial"
                tmp.n,               # number of observations in training data
                p,               # dimension of x
                tmp.matX,         # p x n training data matX
                tmp.Y,            # 1 x n training data Y
                ntree,
                numcut,
                ndpost*keepevery,
                nskip,
                keepevery,
                power,
                base,
                tmp.offsetY,
                tau,
                nu,
                lambda,
                sigest,
                sparse,
                theta,
                omega,
                a,
                b,
                rho,
                augment,
                printevery,
                xinfo)

        res$varcount[[k]] <- res.list[[k]]$varcount
        res$varprob[[k]] <- res.list[[k]]$varprob

        # for(q in 1:pY) {
        #   res$varcount[ , q, k] <- res.list[[k]]$varcount[ , q]
        #   res$varcount.mean[k, q] <- res.list[[k]]$varcount.mean[q]
        #   res$varprob[ , q, k] <- res.list[[k]]$varprob[ , q]
        #   res$varprob.mean[k, q] <- res.list[[k]]$varprob.mean[q]
        # }

        # res$offsetY[k] <- res.list[[k]]$offsetY
        # res$rm.const[[k]] <- res.list[[k]]$rm.const
        res$treedraws$trees[[k]] <- res.list[[k]]$treedraws$trees
      }
    }
    #--------------------------------------------------
    res$treedraws$cutpoints <- res.list[[1]]$treedraws$cutpoints
    dimnames(res$varcount.mean)[[2]] <- dimnames(res$varcount[[1]])[[2]]
    dimnames(res$varprob.mean)[[2]] <- dimnames(res$varprob[[1]])[[2]]
    res$rm.const <- res.list[[1]]$rm.const
    res$comp.test <- NULL
  }
  res$proc.time <- proc.time()-ptm
  
  #--------------------------------------------------
  res$hostname <- hostname
  res$offsetY <- offsetY  
  res$typeY <- typeY
  res$ndpost <- ndpost
  res$rm.const <- rm.const
  res$sigest <- sigest
  # res$B_u <- B_u
  # res$u <- u
  attr(res, 'class') <- "BART"
  
  return(res)
}