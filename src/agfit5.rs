use std::f64::{EPSILON, INFINITY};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Solve, Inverse};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

#[derive(Debug)]
pub struct CoxResult {
    pub coefficients: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub p_values: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub log_likelihood: f64,
    pub score: f64,
    pub wald_test: f64,
    pub iterations: i32,
    pub converged: bool,
    pub variance_matrix: Vec<Vec<f64>>,
}

struct CoxState {
    covar: Vec<Vec<f64>>,
    cmat: Vec<Vec<f64>>,
    cmat2: Vec<Vec<f64>>,
    a: Vec<f64>,
    oldbeta: Vec<f64>,
    a2: Vec<f64>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    event: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    tmean: Vec<f64>,
    ptype: i32,
    pdiag: i32,
    ipen: Vec<f64>,
    upen: Vec<f64>,
    zflag: Vec<i32>,
    frail: Vec<i32>,
    score: Vec<f64>,
    strata: Vec<i32>,
}

impl CoxState {
    fn new(
        nused: usize,
        nvar: usize,
        nfrail: usize,
        yy: &[f64],
        covar2: &[f64],
        offset2: &[f64],
        weights2: &[f64],
        strata: &[i32],
        sort: &[i32],
        ptype: i32,
        pdiag: i32,
        frail2: &[i32],
    ) -> Self {
        let mut covar = vec![vec![0.0; nused]; nvar];
        let mut k = 0;
        for i in 0..nvar {
            for j in 0..nused {
                covar[i][j] = covar2[k];
                k += 1;
            }
        }

        let mut state = CoxState {
            covar,
            cmat: vec![vec![0.0; nvar + 1]; nvar + nfrail],
            cmat2: vec![vec![0.0; nvar + 1]; nvar + nfrail],
            a: vec![0.0; 4 * (nvar + nfrail) + 5 * nused],
            oldbeta: vec![0.0; nvar + nfrail],
            a2: vec![0.0; nvar + nfrail],
            offset: offset2.to_vec(),
            weights: weights2.to_vec(),
            event: yy[2 * nused..3 * nused].iter().map(|&x| x as i32).collect(),
            start: yy[0..nused].to_vec(),
            stop: yy[nused..2 * nused].to_vec(),
            sort1: sort[0..nused].iter().map(|&x| (x - 1) as usize).collect(),
            sort2: sort[nused..2 * nused]
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect(),
            tmean: vec![0.0; nvar + nfrail],
            ptype,
            pdiag,
            ipen: vec![0.0; nfrail.max(nvar * nvar)],
            upen: vec![0.0; nfrail.max(nvar)],
            zflag: vec![0; nvar.max(2)],
            frail: frail2.to_vec(),
            score: vec![0.0; nused],
            strata: strata.to_vec(),
        };

        for i in 0..nvar {
            let mean = state.covar[i].iter().sum::<f64>() / nused as f64;
            for val in &mut state.covar[i] {
                *val -= mean;
            }
        }

        state
    }

    fn update(&mut self, beta: &mut [f64], u: &mut [f64], imat: &mut [f64], loglik: &mut f64) {
        let nvar = beta.len();
        let nfrail = self.frail.len();
        let nvar2 = nvar + nfrail;

        self.a.iter_mut().for_each(|x| *x = 0.0);
        self.a2.iter_mut().for_each(|x| *x = 0.0);
        u.iter_mut().for_each(|x| *x = 0.0);
        imat.iter_mut().for_each(|x| *x = 0.0);

        for person in 0..self.weights.len() {
            let mut zbeta = self.offset[person];
            for i in 0..nvar {
                zbeta += beta[i] * self.covar[i][person];
            }
            if nfrail > 0 {
                zbeta += beta[nvar] * self.frail[person] as f64;
            }
            self.score[person] = zbeta;
        }

        *loglik = 0.0;
        let mut istrat = 0;
        let mut indx2 = 0;

        while istrat < self.strata.len() {
            let mut denom = 0.0;
            let mut risk_sum = 0.0;
            let mut risk_sum2 = 0.0;
            
            for person in istrat..self.weights.len() {
                if self.strata[person] != self.strata[istrat] {
                    break;
                }
                let risk_score = self.score[person].exp();
                risk_sum += self.weights[person] * risk_score;
                risk_sum2 += self.weights[person] * risk_score * risk_score;
            }

            for person in istrat..self.weights.len() {
                if self.strata[person] != self.strata[istrat] {
                    break;
                }
                
                if self.event[person] == 1 {
                    *loglik += self.weights[person] * self.score[person];
                    *loglik -= self.weights[person] * risk_sum.ln();
                    
                    for i in 0..nvar {
                        let mut temp = 0.0;
                        for j in person..self.weights.len() {
                            if self.strata[j] == self.strata[person] {
                                temp += self.weights[j] * self.score[j].exp() * self.covar[i][j];
                            }
                        }
                        u[i] += self.weights[person] * (self.covar[i][person] - temp / risk_sum);
                    }
                    
                    if nfrail > 0 {
                        let mut temp = 0.0;
                        for j in person..self.weights.len() {
                            if self.strata[j] == self.strata[person] {
                                temp += self.weights[j] * self.score[j].exp() * self.frail[j] as f64;
                            }
                        }
                        u[nvar] += self.weights[person] * (self.frail[person] as f64 - temp / risk_sum);
                    }
                    
                    for i in 0..nvar {
                        for j in i..nvar {
                            let mut temp = 0.0;
                            for k in person..self.weights.len() {
                                if self.strata[k] == self.strata[person] {
                                    temp += self.weights[k] * self.score[k].exp() * 
                                           self.covar[i][k] * self.covar[j][k];
                                }
                            }
                            let idx = i * nvar2 + j;
                            imat[idx] += self.weights[person] * 
                                       (temp / risk_sum - 
                                        (self.a[i] * self.a[j]) / (risk_sum * risk_sum));
                        }
                    }
                    
                    if nfrail > 0 {
                        for i in 0..nvar {
                            let mut temp = 0.0;
                            for k in person..self.weights.len() {
                                if self.strata[k] == self.strata[person] {
                                    temp += self.weights[k] * self.score[k].exp() * 
                                           self.covar[i][k] * self.frail[k] as f64;
                                }
                            }
                            let idx = i * nvar2 + nvar;
                            imat[idx] += self.weights[person] * 
                                       (temp / risk_sum - 
                                        (self.a[i] * self.a[nvar]) / (risk_sum * risk_sum));
                        }
                        
                        let mut temp = 0.0;
                        for k in person..self.weights.len() {
                            if self.strata[k] == self.strata[person] {
                                temp += self.weights[k] * self.score[k].exp() * 
                                       (self.frail[k] as f64).powi(2);
                            }
                        }
                        let idx = nvar * nvar2 + nvar;
                        imat[idx] += self.weights[person] * 
                                   (temp / risk_sum - 
                                    (self.a[nvar] * self.a[nvar]) / (risk_sum * risk_sum));
                    }
                }
                
                if person < self.weights.len() - 1 && self.strata[person + 1] == self.strata[person] {
                    let risk_score = self.score[person].exp();
                    risk_sum -= self.weights[person] * risk_score;
                    risk_sum2 -= self.weights[person] * risk_score * risk_score;
                }
            }
            
            while istrat < self.strata.len() && self.strata[istrat] == self.strata[istrat] {
                istrat += 1;
            }
        }
    }
}

pub fn agfit5(
    nused: usize,
    nvar: usize,
    nfrail: usize,
    yy: &[f64],
    covar: &[f64],
    offset: &[f64],
    weights: &[f64],
    strata: &[i32],
    sort: &[i32],
    ptype: i32,
    pdiag: i32,
    frail: &[i32],
    max_iter: i32,
    eps: f64,
) -> Result<CoxResult, Box<dyn std::error::Error>> {
    let mut state = CoxState::new(
        nused, nvar, nfrail, yy, covar, offset, weights, 
        strata, sort, ptype, pdiag, frail
    );
    
    let nvar2 = nvar + nfrail;
    let mut beta = vec![0.0; nvar2];
    let mut u = vec![0.0; nvar2];
    let mut imat = vec![0.0; nvar2 * nvar2];
    let mut loglik = 0.0;
    
    let mut iter = 0;
    let mut converged = false;
    
    while iter < max_iter {
        let old_loglik = loglik;
        
        state.update(&mut beta, &mut u, &mut imat, &mut loglik);
        
        if (loglik - old_loglik).abs() < eps {
            converged = true;
            break;
        }
        
        let mut imat_array = Array2::from_shape_vec((nvar2, nvar2), imat.clone())?;
        let u_array = Array1::from_vec(u.clone());
        
        for i in 0..nvar2 {
            imat_array[[i, i]] += 1e-8;
        }
        
        match imat_array.solve_into(u_array) {
            Ok(delta) => {
                for i in 0..nvar2 {
                    beta[i] += delta[i];
                }
            }
            Err(_) => {
                return Err("Failed to solve linear system".into());
            }
        }
        
        iter += 1;
    }
    
    state.update(&mut beta, &mut u, &mut imat, &mut loglik);
    
    let mut variance_matrix = vec![vec![0.0; nvar2]; nvar2];
    let imat_array = Array2::from_shape_vec((nvar2, nvar2), imat)?;
    
    match imat_array.inv() {
        Ok(inv_imat) => {
            for i in 0..nvar2 {
                for j in 0..nvar2 {
                    variance_matrix[i][j] = inv_imat[[i, j]];
                }
            }
        }
        Err(_) => {
            return Err("Failed to invert information matrix".into());
        }
    }
    
    let standard_errors: Vec<f64> = (0..nvar2)
        .map(|i| (variance_matrix[i][i] as f64).sqrt())
        .collect();
    
    let p_values: Vec<f64> = (0..nvar2)
        .map(|i| {
            if standard_errors[i] > 0.0 {
                let z = beta[i] / standard_errors[i];
                2.0 * (1.0 - normal_cdf(z.abs()))
            } else {
                1.0
            }
        })
        .collect();
    
    let confidence_intervals: Vec<(f64, f64)> = (0..nvar2)
        .map(|i| {
            let se = standard_errors[i];
            let coef = beta[i];
            (coef - 1.96 * se, coef + 1.96 * se)
        })
        .collect();
    
    let score: f64 = u.iter().map(|&x| x * x).sum();
    
    let wald_test: f64 = beta.iter()
        .zip(standard_errors.iter())
        .map(|(&coef, &se)| if se > 0.0 { (coef / se).powi(2) } else { 0.0 })
        .sum();
    
    Ok(CoxResult {
        coefficients: beta,
        standard_errors,
        p_values,
        confidence_intervals,
        log_likelihood: loglik,
        score,
        wald_test,
        iterations: iter,
        converged,
        variance_matrix,
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

#[pyfunction]
pub fn perform_cox_regression_frailty(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    offset: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    frail: Option<Vec<i32>>,
    max_iter: Option<i32>,
    eps: Option<f64>,
) -> PyResult<PyObject> {
    let nused = time.len();
    if nused == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }
    let nvar = covariates.len();
    if nvar == 0 {
        return Err(PyRuntimeError::new_err("No covariates provided"));
    }
    if event.len() != nused {
        return Err(PyRuntimeError::new_err("Event vector length does not match time vector"));
    }
    for cov in &covariates {
        if cov.len() != nused {
            return Err(PyRuntimeError::new_err("Covariate vector length does not match time vector"));
        }
    }
    let offset = offset.unwrap_or_else(|| vec![0.0; nused]);
    let weights = weights.unwrap_or_else(|| vec![1.0; nused]);
    let strata = strata.unwrap_or_else(|| vec![1; nused]);
    let frail = frail.unwrap_or_else(|| vec![0; nused]);
    let max_iter = max_iter.unwrap_or(20);
    let eps = eps.unwrap_or(1e-6);
    let mut yy = Vec::with_capacity(3 * nused);
    yy.extend_from_slice(&time);
    yy.extend_from_slice(&time);
    yy.extend(event.iter().map(|&x| x as f64));
    let mut covar = Vec::with_capacity(nvar * nused);
    for i in 0..nused {
        for j in 0..nvar {
            covar.push(covariates[j][i]);
        }
    }
    let sort: Vec<i32> = (1..=nused as i32).collect();
    let nfrail = if frail.iter().any(|&x| x != 0) { 1 } else { 0 };
    match agfit5(
        nused, nvar, nfrail, &yy, &covar, &offset, &weights,
        &strata, &sort, 0, 0, &frail, max_iter, eps
    ) {
        Ok(result) => {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("coefficients", result.coefficients).unwrap();
                dict.set_item("standard_errors", result.standard_errors).unwrap();
                dict.set_item("p_values", result.p_values).unwrap();
                dict.set_item("confidence_intervals", result.confidence_intervals).unwrap();
                dict.set_item("log_likelihood", result.log_likelihood).unwrap();
                dict.set_item("score", result.score).unwrap();
                dict.set_item("wald_test", result.wald_test).unwrap();
                dict.set_item("iterations", result.iterations).unwrap();
                dict.set_item("converged", result.converged).unwrap();
                dict.set_item("variance_matrix", result.variance_matrix).unwrap();
                Ok(dict.into())
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Cox regression failed: {}", e))),
    }
}
