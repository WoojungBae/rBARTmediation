/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Robert Gramacy
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include <RcppArmadillo.h>
#include "mvnorm.h"

#ifndef NoRcpp

RcppExport SEXP crmvnorm(SEXP n, SEXP mean, SEXP vcov) {
  // BEGIN_RCPP
  // Rcpp::RObject rcpp_result_gen;
  // Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter< arma::uword >::type N(n);
  Rcpp::traits::input_parameter< arma::vec >::type MU(mean);
  Rcpp::traits::input_parameter< arma::mat >::type SIG(vcov);
  return Rcpp::wrap(rmvnorm(N, MU, SIG));
  // END_RCPP
}

#endif
