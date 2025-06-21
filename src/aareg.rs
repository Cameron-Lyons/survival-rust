use ndarray::{Array1, Array2, Axis};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;
use ndarray_linalg::{Solve, Lapack};

#[pyclass]
#[derive(Clone)]
struct AaregOptions {
    #[pyo3(get, set)]
    formula: String,                   // Formula as a string
    #[pyo3(get, set)]
    data: Vec<Vec<f64>>,               // 2D array for the dataset
    #[pyo3(get, set)]
    variable_names: Vec<String>,       // Names of the variables (columns in data)
    #[pyo3(get, set)]
    weights: Option<Vec<f64>>,         // Optional weights
    #[pyo3(get, set)]
    subset: Option<Vec<usize>>,        // Optional subset of data indices
    #[pyo3(get, set)]
    na_action: Option<String>,         // Action for NA values ("Fail" or "Exclude")
    #[pyo3(get, set)]
    qrtol: f64,                        // Tolerance for singularity detection
    #[pyo3(get, set)]
    nmin: Option<usize>,               // Minimum number of observations
    #[pyo3(get, set)]
    dfbeta: bool,                      // Whether to compute dfbeta residuals
    #[pyo3(get, set)]
    taper: f64,                        // Taper parameter
    #[pyo3(get, set)]
    test: Vec<String>,                 // List of tests to perform
    #[pyo3(get, set)]
    cluster: Option<HashMap<String, i32>>, // Optional clustering
    #[pyo3(get, set)]
    model: bool,                       // Whether to include the model frame in the output
    #[pyo3(get, set)]
    x: bool,                           // Whether to include the matrix of predictors
    #[pyo3(get, set)]
    y: bool,                           // Whether to include the response vector
}

#[pymethods]
impl AaregOptions {
    #[new]
    fn new(formula: String, data: Vec<Vec<f64>>, variable_names: Vec<String>) -> Self {
        AaregOptions {
            formula,
            data,
            variable_names,
            weights: None,
            subset: None,
            na_action: None,
            qrtol: 1e-07,
            nmin: None,
            dfbeta: false,
            taper: 1.0,
            test: vec![],
            cluster: None,
            model: false,
            x: false,
            y: false,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct AaregResult {
    #[pyo3(get, set)]
    coefficients: Vec<f64>,                  // Estimated coefficients
    #[pyo3(get, set)]
    standard_errors: Vec<f64>,               // Standard errors
    #[pyo3(get, set)]
    confidence_intervals: Vec<ConfidenceInterval>, // Confidence intervals
    #[pyo3(get, set)]
    p_values: Vec<f64>,                      // P-values
    #[pyo3(get, set)]
    goodness_of_fit: f64,                    // Goodness-of-fit statistic
    #[pyo3(get, set)]
    fit_details: Option<FitDetails>,         // Fit details
    #[pyo3(get, set)]
    residuals: Option<Vec<f64>>,             // Residuals
    #[pyo3(get, set)]
    diagnostics: Option<Diagnostics>,        // Diagnostics
}

#[pyclass]
#[derive(Clone)]
struct ConfidenceInterval {
    #[pyo3(get, set)]
    lower_bound: f64,
    #[pyo3(get, set)]
    upper_bound: f64,
}

#[pyclass]
#[derive(Clone)]
struct FitDetails {
    #[pyo3(get, set)]
    iterations: u32,                  // Number of iterations
    #[pyo3(get, set)]
    converged: bool,                  // Convergence status
    #[pyo3(get, set)]
    final_objective_value: f64,       // Final objective value
    #[pyo3(get, set)]
    convergence_threshold: f64,       // Convergence threshold
    #[pyo3(get, set)]
    change_in_objective: Option<f64>, // Change in objective
    #[pyo3(get, set)]
    max_iterations: Option<u32>,      // Max iterations allowed
    #[pyo3(get, set)]
    optimization_method: Option<String>, // Optimization method used
    #[pyo3(get, set)]
    warnings: Vec<String>,            // Warnings during fitting
}

#[pyclass]
#[derive(Clone)]
struct Diagnostics {
    #[pyo3(get, set)]
    dfbetas: Option<Vec<f64>>,            // DFBetas
    #[pyo3(get, set)]
    cooks_distance: Option<Vec<f64>>,     // Cook's distance
    #[pyo3(get, set)]
    leverage: Option<Vec<f64>>,           // Leverage values
    #[pyo3(get, set)]
    deviance_residuals: Option<Vec<f64>>, // Deviance residuals
    #[pyo3(get, set)]
    martingale_residuals: Option<Vec<f64>>, // Martingale residuals
    #[pyo3(get, set)]
    schoenfeld_residuals: Option<Vec<f64>>, // Schoenfeld residuals
    #[pyo3(get, set)]
    score_residuals: Option<Vec<f64>>,    // Score residuals
    #[pyo3(get, set)]
    additional_measures: Option<Vec<f64>>, // Additional measures
}

#[derive(Debug)]
enum AaregError {
    DataError(String),
    FormulaError(String),
    WeightsError(String),
    CalculationError(String),
    InputError(String),
    InternalError(String),
    GenericError(String),
}

impl fmt::Display for AaregError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AaregError::DataError(msg) => write!(f, "Data Error: {}", msg),
            AaregError::FormulaError(msg) => write!(f, "Formula Error: {}", msg),
            AaregError::WeightsError(msg) => write!(f, "Weights Error: {}", msg),
            AaregError::CalculationError(msg) => write!(f, "Calculation Error: {}", msg),
            AaregError::InputError(msg) => write!(f, "Input Error: {}", msg),
            AaregError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
            AaregError::GenericError(msg) => write!(f, "Generic Error: {}", msg),
        }
    }
}

impl From<pyo3::PyErr> for AaregError {
    fn from(err: pyo3::PyErr) -> AaregError {
        AaregError::GenericError(err.to_string())
    }
}

impl From<AaregError> for PyErr {
    fn from(err: AaregError) -> PyErr {
        PyRuntimeError::new_err(format!("Aareg error: {}", err))
    }
}

#[pyfunction]
fn aareg(options: AaregOptions) -> PyResult<AaregResult> {
    let data_array = Array2::from_shape_vec(
        (options.data.len(), options.data[0].len()),
        options.data.clone().into_iter().flatten().collect(),
    )
    .map_err(|e| AaregError::DataError(e.to_string()))?;

    let (response_name, covariate_names) = parse_formula(&options.formula)?;

    let subset_data = apply_subset(&data_array, &options.subset)?;

    let weighted_data = apply_weights(&subset_data, options.weights.clone())?;

    let filtered_data = handle_missing_data(&weighted_data, options.na_action.clone())?;

    let (y, x) = prepare_data_for_regression(
        &filtered_data,
        &response_name,
        &covariate_names,
        &options.variable_names,
    )?;

    let regression_result = perform_aalen_regression(&y, &x, &options)?;

    let processed_result = post_process_results(regression_result, &options)?;

    Ok(processed_result)
}

fn parse_formula(formula: &str) -> Result<(String, Vec<String>), AaregError> {
    let mut formula_parts = formula.splitn(2, '~');
    let response = formula_parts
        .next()
        .ok_or_else(|| AaregError::FormulaError("Formula is missing a response variable.".to_string()))?
        .trim()
        .to_string();
    let covariates_str = formula_parts
        .next()
        .ok_or_else(|| AaregError::FormulaError("Formula is missing covariates.".to_string()))?
        .trim();
    let covariates = covariates_str
        .split('+')
        .map(|s| s.trim().to_string())
        .collect();
    Ok((response, covariates))
}

fn apply_subset(
    data: &Array2<f64>,
    subset: &Option<Vec<usize>>,
) -> Result<Array2<f64>, AaregError> {
    match subset {
        Some(s) => {
            if s.iter().any(|&i| i >= data.nrows()) {
                return Err(AaregError::DataError(
                    "Subset indices are out of bounds".to_string(),
                ));
            }
            let subset_data = data.select(Axis(0), s);
            Ok(subset_data)
        }
        None => Ok(data.clone()),
    }
}

fn apply_weights(
    data: &Array2<f64>,
    weights: Option<Vec<f64>>,
) -> Result<Array2<f64>, AaregError> {
    match weights {
        Some(w) => {
            if w.len() != data.nrows() {
                return Err(AaregError::WeightsError(
                    "Weights length does not match number of observations".to_string(),
                ));
            }
            let weights_array = Array1::from_vec(w);
            let weighted_data = data * &weights_array.insert_axis(Axis(1));
            Ok(weighted_data)
        }
        None => Ok(data.clone()),
    }
}

fn handle_missing_data(
    data: &Array2<f64>,
    na_action: Option<String>,
) -> Result<Array2<f64>, AaregError> {
    match na_action.as_deref() {
        Some("Fail") => {
            if data.iter().any(|x| x.is_nan()) {
                Err(AaregError::InputError(
                    "Invalid input: missing values in data".to_string(),
                ))
            } else {
                Ok(data.clone())
            }
        }
        Some("Exclude") => {
            let not_nan_rows: Vec<usize> = data
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(_, row)| !row.iter().any(|x| x.is_nan()))
                .map(|(i, _)| i)
                .collect();

            if not_nan_rows.is_empty() {
                Err(AaregError::InputError(
                    "All rows contain NaN values".to_string(),
                ))
            } else {
                Ok(data.select(Axis(0), &not_nan_rows))
            }
        }
        Some(other) => Err(AaregError::InputError(format!(
            "Invalid na_action '{}'. Expected 'Fail' or 'Exclude'.",
            other
        ))),
        None => Ok(data.clone()),
    }
}

fn prepare_data_for_regression(
    data: &Array2<f64>,
    response_name: &String,
    covariate_names: &[String],
    variable_names: &[String],
) -> Result<(Array1<f64>, Array2<f64>), AaregError> {
    let mut name_to_index = HashMap::new();
    for (i, name) in variable_names.iter().enumerate() {
        name_to_index.insert(name.clone(), i);
    }

    let response_index = name_to_index
        .get(response_name)
        .ok_or_else(|| AaregError::FormulaError(format!("Response variable '{}' not found.", response_name)))?;

    let mut covariate_indices = Vec::new();
    for cov_name in covariate_names {
        let idx = name_to_index.get(cov_name).ok_or_else(|| {
            AaregError::FormulaError(format!("Covariate '{}' not found.", cov_name))
        })?;
        covariate_indices.push(*idx);
    }

    let y = data.column(*response_index).to_owned();
    let x = data.select(Axis(1), &covariate_indices);

    Ok((y, x))
}

fn perform_aalen_regression(
    y: &Array1<f64>,
    x: &Array2<f64>,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    let n = y.len();
    let p = x.ncols();
    
    if n == 0 || p == 0 {
        return Err(AaregError::DataError("Empty dataset or no covariates".to_string()));
    }
    
    if n < p {
        return Err(AaregError::DataError("More covariates than observations".to_string()));
    }
    
    if let Some(nmin) = options.nmin {
        if n < nmin {
            return Err(AaregError::DataError(
                format!("Number of observations ({}) is less than minimum required ({})", n, nmin)
            ));
        }
    }
    
    let mut design_matrix = Array2::zeros((n, p + 1));
    design_matrix.column_mut(0).fill(1.0); // Intercept
    for j in 0..p {
        design_matrix.column_mut(j + 1).assign(&x.column(j));
    }
    
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
    
    let sorted_times: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    
    let mut sorted_design = Array2::zeros((n, p + 1));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for j in 0..(p + 1) {
            sorted_design[[new_idx, j]] = design_matrix[[old_idx, j]];
        }
    }
    
    let mut coefficients = vec![0.0; p + 1];
    let mut cumulative_coefficients = vec![0.0; p + 1];
    let mut standard_errors = vec![0.0; p + 1];
    let mut p_values = vec![1.0; p + 1];
    
    let mut unique_times = Vec::new();
    let mut time_indices = Vec::new();
    
    for (i, &time) in sorted_times.iter().enumerate() {
        if unique_times.is_empty() || ((time - unique_times.last().unwrap()) as f64).abs() > 1e-10 {
            unique_times.push(time);
            time_indices.push(i);
        }
    }
    
    let num_unique_times = unique_times.len();
    
    let mut warnings = Vec::new();
    let mut converged = true;
    let mut iterations = 0;
    let max_iterations = 100;
    
    for t_idx in 0..num_unique_times {
        let current_time = unique_times[t_idx];
        let event_idx = time_indices[t_idx];
        
        let at_risk: Vec<usize> = (event_idx..n).collect();
        
        if at_risk.len() < p + 1 {
            warnings.push(format!("Insufficient observations at risk at time {:.3}", current_time));
            continue;
        }
        
        let at_risk_design = sorted_design.select(Axis(0), &at_risk);
        
        let xtx = at_risk_design.t().dot(&at_risk_design);
        
        let dN = vec![1.0; at_risk.len()]; // Assuming all events are failures
        let xt_dn = at_risk_design.t().dot(&Array1::from_vec(dN.clone()));

        let beta_increment = xtx
            .solve_into(xt_dn)
            .map_err(|_| AaregError::CalculationError("Failed to solve linear system".to_string()))?;
        
        for i in 0..coefficients.len() {
            cumulative_coefficients[i] += beta_increment[i];
        }
        
        let residuals = at_risk_design.dot(&beta_increment) - &Array1::from_vec(dN);
        let residual_variance = residuals.dot(&residuals) / (at_risk.len() as f64 - p as f64 - 1.0);
        
        for i in 0..standard_errors.len() {
            let se: f64 = standard_errors[i];
            standard_errors[i] = (se.powi(2) + residual_variance).sqrt();
        }
        
        iterations += 1;
        if iterations >= max_iterations {
            converged = false;
            warnings.push("Maximum iterations reached".to_string());
            break;
        }
    }
    
    coefficients = cumulative_coefficients.clone();
    
    for i in 0..p_values.len() {
        if standard_errors[i] > 0.0 {
            let z_stat: f64 = coefficients[i] / standard_errors[i];
            p_values[i] = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
        }
    }
    
    let confidence_intervals: Vec<ConfidenceInterval> = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&coef, &se)| {
            let margin = 1.96 * se; // 95% confidence interval
            ConfidenceInterval {
                lower_bound: coef - margin,
                upper_bound: coef + margin,
            }
        })
        .collect();
    
    let mean_y = y.mean().unwrap_or(0.0);
    let total_ss: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let predicted = design_matrix.dot(&Array1::from_vec(coefficients.clone()));
    let residual_ss: f64 = y.iter().zip(predicted.iter()).map(|(&yi, &pi)| (yi - pi).powi(2)).sum();
    let goodness_of_fit = if total_ss > 0.0 { 1.0 - residual_ss / total_ss } else { 0.0 };
    
    let residuals: Vec<f64> = y.iter().zip(predicted.iter()).map(|(&yi, &pi)| yi - pi).collect();
    
    Ok(AaregResult {
        coefficients,
        standard_errors,
        confidence_intervals,
        p_values,
        goodness_of_fit,
        fit_details: Some(FitDetails {
            iterations,
            converged,
            final_objective_value: residual_ss,
            convergence_threshold: options.qrtol,
            change_in_objective: None,
            max_iterations: Some(max_iterations as u32),
            optimization_method: Some("Aalen's Additive Regression".to_string()),
            warnings,
        }),
        residuals: Some(residuals),
        diagnostics: Some(Diagnostics {
            dfbetas: None,
            cooks_distance: None,
            leverage: None,
            deviance_residuals: None,
            martingale_residuals: None,
            schoenfeld_residuals: None,
            score_residuals: None,
            additional_measures: None,
        }),
    })
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

fn post_process_results(
    mut regression_result: AaregResult,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    if options.dfbeta {
        if let Some(ref mut diagnostics) = regression_result.diagnostics {
            diagnostics.dfbetas = Some(vec![0.0; regression_result.coefficients.len()]);
        }
    }
    Ok(regression_result)
}
