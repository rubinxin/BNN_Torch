import numpy as np
from scipy import stats

standard_norm = stats.norm(loc=0, scale=1)

def calculate_residuals_density(normalized_residuals, percentile):
    '''
    Calculate the fraction of the residuals that fall within the lower
    `percentile` of their respective Gaussian distributions, which are
    defined by their respective uncertainty estimates.
    '''
    # Find the normalized bounds of this percentile
    upper_bound = standard_norm.ppf(percentile)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the empirical fraction of residuals that fall within the bounds
    density = num_within_quantile / len(normalized_residuals)
    return density


def calibration_error_regression(pred_mean, pred_std, y_test):
    predicted_pi = np.linspace(0, 1, 100)
    # Normalize the residuals so they all should fall on the normal bell curve
    residuals = pred_mean - y_test
    normalized_residuals = residuals / pred_std

    observed_pi = np.array([calculate_residuals_density(normalized_residuals, quantile)
                   for quantile in predicted_pi])
    calibration_error = ((predicted_pi - observed_pi) ** 2).sum()
    return calibration_error, predicted_pi, observed_pi


def nll(pred_mean, pred_std, y_test):
    return - np.mean(stats.norm.logpdf(y_test, loc=pred_mean, scale=pred_std))

