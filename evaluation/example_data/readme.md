# Example data provided for evaluation.

The number in the filename indicates which NHM cutout model results are in the file. "Prior" indicates this is prior Monte Carlo results.

The index column is the observation name which can contain some metadata. The metadata are delimited by ":" and the first item is typically also the descriptive group name. Second is time (either month, year, or year_month) and the final item is the location (an HRU or a segment).

The modelled and measured columns are, as indicated, the model or observation values in appropriate units. In this case, the measurement value has not been perturbed with noise, but other ensemble members are subjected to noise.

The group column describes the data type. Those beginning in "g_min" or "l_max" are inequality observations in which weight is only assigned if the inequality is violated - these are used to enforce ranges without targeting specific values.

The weight is the weight assigned in the regression when forming the weighted sum of squares objective function.

The standard deviation (which, ideally, should be the reciprocal of weight, but the weight is adjusted thus violating this) is used to sample i.i.d. noise around measurement values for the ensemble.
