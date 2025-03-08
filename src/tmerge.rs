pub fn tmerge(
    id: &[i32],
    time1: &[f64],
    newx: &[f64],
    nid: &[i32],
    ntime: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let n1 = id.len();
    let n2 = nid.len();
    let mut result = newx.to_vec();
    let mut k = 0;
    let mut current_id = -1;
    let mut csum = 0.0;
    let mut has_one = false;

    for i in 0..n1 {
        if id[i] != current_id {
            current_id = id[i];
            csum = 0.0;
            has_one = false;
            while k < n2 && nid[k] < current_id {
                k += 1;
            }
        }

        let start_time = time1[i];
        let mut local_k = k;

        while local_k < n2 && nid[local_k] == current_id && ntime[local_k] <= start_time {
            csum += x[local_k];
            has_one = true;
            local_k += 1;
        }

        if has_one {
            result[i] = if result[i].is_nan() {
                csum
            } else {
                result[i] + csum
            };
        }
    }

    result
}

pub fn tmerge2(id: &[i32], time1: &[f64], nid: &[i32], ntime: &[f64]) -> Vec<usize> {
    let n1 = id.len();
    let n2 = nid.len();
    let mut result = vec![0; n1];
    let mut k = 0;

    for i in 0..n1 {
        let current_id = id[i];
        let start_time = time1[i];
        result[i] = 0;

        while k < n2 && nid[k] < current_id {
            k += 1;
        }

        let mut last_valid = 0;
        let mut local_k = k;

        while local_k < n2 && nid[local_k] == current_id && ntime[local_k] <= start_time {
            last_valid = local_k + 1; // Convert to 1-based index
            local_k += 1;
        }

        result[i] = last_valid;
    }

    result
}

pub fn tmerge3(id: &[i32], miss: &[bool]) -> Vec<usize> {
    let n = id.len();
    let mut result = vec![0; n];
    let mut last_good = 0;
    let mut current_id = -1;

    for (i, (&current, &is_missing)) in id.iter().zip(miss).enumerate() {
        if current != current_id {
            current_id = current;
            last_good = 0;
        }

        if is_missing {
            result[i] = last_good;
        } else {
            result[i] = i + 1; // 1-based index
            last_good = i + 1;
        }
    }

    result
}

