
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017 Robert McCulloch and Rodney Sparapani

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

predict.rBART <- function(object, newdata, Uindex, ...) {
  call <- prBART
  return(call(object, newdata, Uindex, ...))
  # function(object, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...) {
  # ##if(class(newdata) != "matrix") stop("newdata must be a matrix")
  # 
  # if(.Platform$OS.type == "unix") {
  #   mc.cores.detected <- detectCores()
  # } else {
  #   mc.cores.detected <- NA
  # }
  # 
  # if(!is.na(mc.cores.detected) && mc.cores>mc.cores.detected) {
  #   mc.cores <- mc.cores.detected
  # }
  # 
  # p <- length(object$treedraws$cutpoints)
  # 
  # if(p!=ncol(newdata)) {
  #   stop(paste0('The number of columns in newdata must be equal to ', p))
  # }
  # 
  # if(.Platform$OS.type != "unix" || openmp || mc.cores==1) {
  #   call <- prBART
  # } else {
  #   call <- prBART
  # }
  # 
  # if(object$typeY == "continuous"){
  #   out <- list()
  #   out$pred = call(newdata, object$treedraws, mc.cores=mc.cores, 
  #                   mu=object$offsetY, ...)
  # } else if(object$typeY == "binary"){
  #   out <- list()
  #   out$pred = call(newdata, object$treedraws, mc.cores=mc.cores, 
  #                   mu=object$offsetY, ...)
  #   out$prob <- pnorm(out$pred)
  # } else if(object$typeY == "multinomial"){
  #   kY <- object$kY
  #   lY <- kY-1
  #   pred <- as.list(1:lY)
  #   trees <- object$treedraws$trees
  #   
  #   for(k in 1:lY) {
  #     ## eval(parse(text=paste0('object$treedraws$trees=',
  #     ##                        'object$treedraws$tree', k)))
  #     object$treedraws$trees <- trees[[k]]
  #     pred[[k]] <- list(yhat.test=call(newdata, object$treedraws,
  #                                      mc.cores=mc.cores,
  #                                      mu=object$offset[k], ...))
  #     ## predict.gbart testing
  #     ## pred[[k]] <- call(object, newdata, mc.cores=mc.cores,
  #     ##                   openmp=openmp, type=object$type)
  #   }
  #   H <- dim(pred[[1]]$yhat.test)
  #   ndpost <- H[1]
  #   np <- H[2]
  #   res <- list()
  #   res$yhat.test <- matrix(nrow=ndpost, ncol=kY*np)
  #   res$prob.test <- matrix(nrow=ndpost, ncol=kY*np)
  #   res$comp.test <- matrix(nrow=ndpost, ncol=kY*np)
  #   
  #   for(i in 1:np) {
  #     for(j in 1:kY) {
  #       k <- (i-1)*kY+j
  #       if(j<kY) {
  #         res$yhat.test[ , k] <- pred[[j]]$yhat.test[ , i]
  #         res$prob.test[ , k] <- pnorm(res$yhat.test[ , k])
  #         if(j==1) {
  #           res$comp.test[ , k] <- 1-res$prob.test[ , k]
  #         } else {
  #           res$comp.test[ , k] <- res$comp.test[ , k-1]*
  #             (1-res$prob.test[ , k])
  #           res$prob.test[ , k] <- res$comp.test[ , k-1]*
  #             res$prob.test[ , k]
  #         }
  #       } else {
  #         res$prob.test[ , k] <- res$comp.test[ , k-1]
  #       }
  #     }
  #   }
  #   
  #   res$prob.test.mean <- apply(res$prob.test, 2, mean)
  #   res$yhat.test.mean <- NULL
  #   res$comp.test <- NULL
  #   res$kY <- kY
  #   res$offset <- object$offset
  #   attr(res, 'class') <- 'rBART'
  # }
  # 
  # return(out)
}