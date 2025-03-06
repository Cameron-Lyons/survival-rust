use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

const SMALL: f64 = -200.0;

pub struct SurvivalLikelihood {
    pub loglik: f64,
    pub u: Array1<f64>,
    pub imat: Array2<f64>,
    pub jj: Array2<f64>,
    pub fdiag: Array1<f64>,
    pub jdiag: Array1<f64>,
}

pub type CallbackFunc = fn(z: &[f64], result: &mut [f64]);

pub fn survregc2(
    n: usize,
    nvar: usize,
    nstrat: usize,
    whichcase: bool,
    beta: &ArrayView1<f64>,
    strat: &ArrayView1<i32>,
    offset: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    time2: Option<&ArrayView1<f64>>,
    status: &ArrayView1<i32>,
    wt: &ArrayView1<f64>,
    covar: &ArrayView2<f64>,
    nf: usize,
    frail: &ArrayView1<i32>,
    callback: CallbackFunc,
) -> Result<SurvivalLikelihood, &'static str> {
    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nf;

    let mut result = SurvivalLikelihood {
        loglik: 0.0,
        u: Array1::zeros(nvar3),
        imat: Array2::zeros((nvar2, nvar3)),
        jj: Array2::zeros((nvar2, nvar3)),
        fdiag: Array1::zeros(nf),
        jdiag: Array1::zeros(nf),
    };

    let mut icount = n;
    let mut z = vec![0.0; n * 2]; // Worst case size for interval censoring

    for person in 0..n {
        let strata = if nstrat > 1 {
            (strat[person] - 1) as usize
        } else {
            0
        };

        let sigma = if nstrat > 0 {
            beta[nvar + nf + strata].exp()
        } else {
            beta[nvar + nf].exp()
        };

        let mut eta = offset[person];
        for i in 0..nvar {
            eta += beta[i + nf] * covar[[i, person]];
        }

        if nf > 0 {
            let fgrp = (frail[person] - 1) as usize;
            eta += beta[fgrp];
        }

        z[person] = (time1[person] - eta) / sigma;

        if status[person] == 3 {
            if let Some(t2) = time2 {
                z[icount] = (t2[person] - eta) / sigma;
                icount += 1;
            } else {
                return Err("Missing time2 for interval censored data");
            }
        }
    }

    let mut callback_result = vec![0.0; icount * 5];
    callback(&z[0..icount], &mut callback_result);

    let mut icount = n;
    for person in 0..n {
        let strata = if nstrat > 1 {
            (strat[person] - 1) as usize
        } else {
            0
        };

        let sigma = if nstrat > 0 {
            beta[nvar + nf + strata].exp()
        } else {
            beta[nvar + nf].exp()
        };
        let sig2 = 1.0 / (sigma * sigma);

        let zz = z[person];
        let sz = zz * sigma;
        let status = status[person];

        let (g, dg, ddg, dsig, ddsig, dsg) = match status {
            1 => process_exact(person, sigma, sz, &callback_result),
            0 => process_right_censored(person, sigma, sz, &callback_result),
            2 => process_left_censored(person, sigma, sz, &callback_result),
            3 => process_interval_censored(person, &mut icount, sigma, zz, &z, &callback_result),
            _ => return Err("Invalid status code"),
        }?;

        result.loglik += g * wt[person];
        if whichcase {
            continue;
        }

        update_derivatives(
            &mut result,
            person,
            strata,
            nvar,
            nf,
            frail,
            covar,
            wt[person],
            dg,
            ddg,
            dsig,
            ddsig,
            dsg,
            sigma,
            sz,
        );
    }

    Ok(result)
}

fn process_exact(
    person: usize,
    sigma: f64,
    sz: f64,
    callback_result: &[f64],
) -> Result<(f64, f64, f64, f64, f64, f64), &'static str> {
    let f = callback_result[person * 5 + 2];
    if f <= 0.0 {
        Ok((SMALL, -sz / sigma, -1.0 / sigma, 0.0, 0.0, 0.0))
    } else {
        let df = callback_result[person * 5 + 3];
        let ddf = callback_result[person * 5 + 4];

        let g = f.ln() - sigma.ln();
        let temp1 = df / sigma;
        let temp2 = ddf * sigma.powi(-2);

        let dg = -temp1;
        let dsig = -(sz * temp1 + 1.0);
        let ddg = temp2 - dg.powi(2);
        let dsg = sz * temp2 - dg * (1.0 - sz * temp1);
        let ddsig = sz.powi(2) * temp2 + sz * temp1 * (1.0 - sz * temp1);

        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}

fn update_derivatives(
    res: &mut SurvivalLikelihood,
    person: usize,
    strata: usize,
    nvar: usize,
    nf: usize,
    frail: &ArrayView1<i32>,
    covar: &ArrayView2<f64>,
    weight: f64,
    dg: f64,
    ddg: f64,
    dsig: f64,
    ddsig: f64,
    dsg: f64,
    sigma: f64,
    sz: f64,
) {
    if nf > 0 {
        let fgrp = (frail[person] - 1) as usize;
        res.u[fgrp] += dg * weight;
        res.fdiag[fgrp] -= ddg * weight;
        res.jdiag[fgrp] += dg.powi(2) * weight;
    }

    for i in 0..nvar {
        let cov_i = covar[[i, person]];
        let temp = dg * cov_i * weight;
        res.u[i + nf] += temp;

        for j in 0..=i {
            let cov_j = covar[[j, person]];
            res.imat[[i, j + nf]] -= cov_i * cov_j * ddg * weight;
            res.jj[[i, j + nf]] += temp * cov_j * dg;
        }

        if nf > 0 {
            let fgrp = (frail[person] - 1) as usize;
            res.imat[[i, fgrp]] -= cov_i * ddg * weight;
            res.jj[[i, fgrp]] += temp * dg;
        }
    }

    if nstrat > 0 {
        let k = strata + nvar;
        res.u[k + nf] += dsig * weight;

        for i in 0..nvar {
            let cov_i = covar[[i, person]];
            res.imat[[k, i + nf]] -= dsg * cov_i * weight;
            res.jj[[k, i + nf]] += dsig * cov_i * dg * weight;
        }

        res.imat[[k, k + nf]] -= ddsig * weight;
        res.jj[[k, k + nf]] += dsig.powi(2) * weight;

        if nf > 0 {
            let fgrp = (frail[person] - 1) as usize;
            res.imat[[k, fgrp]] -= dsg * weight;
            res.jj[[k, fgrp]] += dsig * dg * weight;
        }
    }
}
