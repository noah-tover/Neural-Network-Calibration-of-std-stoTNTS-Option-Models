This repository contains my code for my research under Dr. Aaron YS Kim. This research seeks to accomplish three main goals:

* Use Monte Carlo Simulation of the ARMA-GARCH-NTS and Stochcastic Normal Tempered Stable ARMA-GARCH option pricing models to create a map of parameter vectors to their resulting call and put prices.
* Develop a host of candidate surrogate models capable of quickly and accurately predicting the resulting call or put price from a parameter vector. Neural networks and grid interpolation are of particular interest due to their ability to maintain arbitrage free predictions.
* Compare the performance of these surrogates on real world option data to the Monte Carlo method. 

