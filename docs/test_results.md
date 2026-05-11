# Test Results

This document should be updated after each formal test run. The current entries summarize the most recent automated verification performed during development.

## Run Summary

- Test command: `pytest -q`
- Status: passing
- Last observed result: `10 passed`
- Notes: the suite currently emits pandas deprecation warnings from the environment, but no test failures were observed.

## Results Table

| Test Name | Expected Result | Actual Result | Pass/Fail | Interpretation | Bugs Found and Fixed |
| --- | --- | --- | --- | --- | --- |
| Black-Scholes put-call parity | Approximate parity holds within tolerance | Passed | Pass | Pricing formula is consistent. | None |
| Input validation | Invalid inputs raise `ValueError` | Passed | Pass | Input checks reject malformed portfolio files. | None |
| Portfolio valuation | Returns numeric portfolio value | Passed | Pass | Portfolio aggregation works. | None |
| Covariance matrix dimensions | Matrix shape matches number of assets | Passed | Pass | Calibration output is dimensionally consistent. | None |
| VaR non-negativity | VaR is non-negative | Passed | Pass | All three model families return non-negative VaR. | None |
| Historical scenario repricing | Repricing uses historical log-return shocks | Passed | Pass | Historical VaR uses full revaluation under historical scenarios. | None |
| Monte Carlo reproducibility | Same seed yields same result | Passed | Pass | Random seed is fixed for deterministic simulation output. | None |
| Backtesting exception count | Exception count is reported correctly | Passed | Pass | Backtesting summary reports exception counts and rates. | None |
| No lookahead bias | Backtest only uses data up to calibration date | Passed | Pass | Rolling calibration window uses only past data. | None |

### Automated Test Suite Result

The pytest suite completed successfully.

10 passed in 10.38s

### Historical, MC, and Parametric VaR are very close

Historical VaR: 1300.05
Monte Carlo VaR: 1300.31
Parametric VaR: 1300.80

The three VaR methodologies produced very similar results for the sample portfolio. This suggests that the portfolio risk is dominated by approximately linear stock exposure and that the option positions are not large enough to generate substantial nonlinear effects over a one-day horizon.

## Update Instructions

When new tests are added or the suite changes:

1. Run `pytest -q` from the project root.
2. Record the command and total pass/fail count above.
3. Update any affected rows in the results table.
4. Document any bugs that were discovered and fixed during testing.

## Known Issues

- The environment currently emits pandas deprecation warnings during test execution.
- These warnings do not affect the current grading workflow, but they should be reviewed in a future cleanup pass.

