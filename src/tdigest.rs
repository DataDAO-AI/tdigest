use std::cmp::Ordering;

const DEFAULT_MAX_SIZE: usize = 100;

/// Centroid implementation to the cluster mentioned in the paper.
#[derive(Debug, PartialEq, Clone)]
struct Centroid {
  mean: f64,
  weight: f64,
}

impl PartialOrd for Centroid {
  fn partial_cmp(&self, other: &Centroid) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Eq for Centroid {}

impl Ord for Centroid {
  fn cmp(&self, other: &Centroid) -> Ordering {
    self.mean.total_cmp(&other.mean)
  }
}

impl Centroid {
  fn new(mean: f64, weight: f64) -> Self {
    Centroid { mean, weight }
  }

  #[inline]
  fn mean(&self) -> f64 {
    self.mean
  }

  #[inline]
  fn weight(&self) -> f64 {
    self.weight
  }

  fn add(&mut self, sum: f64, weight: f64) -> f64 {
    let new_sum = sum + self.weight * self.mean;
    let new_weight = self.weight + weight;
    self.weight = new_weight;
    self.mean = new_sum / new_weight;
    new_sum
  }
}

impl Default for Centroid {
  fn default() -> Self {
    Centroid {
      mean: 0_f64,
      weight: 1_f64,
    }
  }
}

/// T-Digest to be operated on.
#[derive(Debug, PartialEq, Clone)]
pub struct TDigest {
  centroids: Vec<Centroid>,
  max_size: usize,
  sum: f64,
  count: f64,
  max: f64,
  min: f64,
}

impl TDigest {
  pub fn new(max_size: usize) -> Self {
    TDigest {
      centroids: Vec::new(),
      max_size,
      sum: 0_f64,
      count: 0_f64,
      max: f64::NAN,
      min: f64::NAN,
    }
  }

  #[inline]
  pub fn count(&self) -> f64 {
    self.count
  }

  #[inline]
  pub fn max(&self) -> f64 {
    self.max
  }

  #[inline]
  pub fn min(&self) -> f64 {
    self.min
  }

  #[inline]
  pub fn max_size(&self) -> usize {
    self.max_size
  }

  /// Size in bytes including `Self`.
  pub fn size(&self) -> usize {
    std::mem::size_of_val(self)
      + (std::mem::size_of::<Centroid>() * self.centroids.capacity())
  }
}

impl Default for TDigest {
  fn default() -> Self {
    TDigest {
      centroids: Vec::new(),
      max_size: DEFAULT_MAX_SIZE,
      sum: 0_f64,
      count: 0_f64,
      max: f64::NAN,
      min: f64::NAN,
    }
  }
}

impl TDigest {
  fn k_to_q(k: f64, d: f64) -> f64 {
    let k_div_d = k / d;
    if k_div_d >= 0.5 {
      let base = 1.0 - k_div_d;
      1.0 - 2.0 * base * base
    } else {
      2.0 * k_div_d * k_div_d
    }
  }

  fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if lo.is_nan() && hi.is_nan() {
      return v;
    }
    v.clamp(lo, hi)
  }

  #[cfg(test)]
  pub fn merge_unsorted_f64(&self, unsorted_values: Vec<f64>) -> TDigest {
    let mut values = unsorted_values;
    values.sort_by(|a, b| a.total_cmp(b));
    self.merge_sorted_f64(&values)
  }

  pub fn merge_sorted_f64(&self, sorted_values: &[f64]) -> TDigest {
    #[cfg(debug_assertions)]
    debug_assert!(is_sorted(sorted_values), "unsorted input to TDigest");

    if sorted_values.is_empty() {
      return self.clone();
    }

    let mut result = TDigest::new(self.max_size());
    result.count = self.count() + (sorted_values.len() as f64);

    let maybe_min = *sorted_values.first().unwrap();
    let maybe_max = *sorted_values.last().unwrap();

    if self.count() > 0.0 {
      result.min = self.min.min(maybe_min);
      result.max = self.max.max(maybe_max);
    } else {
      result.min = maybe_min;
      result.max = maybe_max;
    }

    let mut compressed: Vec<Centroid> = Vec::with_capacity(self.max_size);

    let mut k_limit: f64 = 1.0;
    let mut q_limit_times_count =
      Self::k_to_q(k_limit, self.max_size as f64) * result.count();
    k_limit += 1.0;

    let mut iter_centroids = self.centroids.iter().peekable();
    let mut iter_sorted_values = sorted_values.iter().peekable();

    let mut curr: Centroid = if let Some(c) = iter_centroids.peek() {
      let curr = **iter_sorted_values.peek().unwrap();
      if c.mean() < curr {
        iter_centroids.next().unwrap().clone()
      } else {
        Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
      }
    } else {
      Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
    };

    let mut weight_so_far = curr.weight();

    let mut sums_to_merge = 0_f64;
    let mut weights_to_merge = 0_f64;

    while iter_centroids.peek().is_some() || iter_sorted_values.peek().is_some()
    {
      let next: Centroid = if let Some(c) = iter_centroids.peek() {
        if iter_sorted_values.peek().is_none()
          || c.mean() < **iter_sorted_values.peek().unwrap()
        {
          iter_centroids.next().unwrap().clone()
        } else {
          Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
        }
      } else {
        Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
      };

      let next_sum = next.mean() * next.weight();
      weight_so_far += next.weight();

      if weight_so_far <= q_limit_times_count {
        sums_to_merge += next_sum;
        weights_to_merge += next.weight();
      } else {
        result.sum += curr.add(sums_to_merge, weights_to_merge);
        sums_to_merge = 0_f64;
        weights_to_merge = 0_f64;

        compressed.push(curr.clone());
        q_limit_times_count =
          Self::k_to_q(k_limit, self.max_size as f64) * result.count();
        k_limit += 1.0;
        curr = next;
      }
    }

    result.sum += curr.add(sums_to_merge, weights_to_merge);
    compressed.push(curr);
    compressed.shrink_to_fit();
    compressed.sort();

    result.centroids = compressed;
    result
  }

  fn external_merge(
    centroids: &mut [Centroid],
    first: usize,
    middle: usize,
    last: usize,
  ) {
    let mut result: Vec<Centroid> = Vec::with_capacity(centroids.len());

    let mut i = first;
    let mut j = middle;

    while i < middle && j < last {
      match centroids[i].cmp(&centroids[j]) {
        Ordering::Less => {
          result.push(centroids[i].clone());
          i += 1;
        }
        Ordering::Greater => {
          result.push(centroids[j].clone());
          j += 1;
        }
        Ordering::Equal => {
          result.push(centroids[i].clone());
          i += 1;
        }
      }
    }

    while i < middle {
      result.push(centroids[i].clone());
      i += 1;
    }

    while j < last {
      result.push(centroids[j].clone());
      j += 1;
    }

    i = first;
    for centroid in result.into_iter() {
      centroids[i] = centroid;
      i += 1;
    }
  }

  // Merge multiple T-Digests
  pub fn merge_digests(digests: &[TDigest]) -> TDigest {
    let n_centroids: usize = digests.iter().map(|d| d.centroids.len()).sum();
    if n_centroids == 0 {
      return TDigest::default();
    }

    let max_size = digests.first().unwrap().max_size;
    let mut centroids: Vec<Centroid> = Vec::with_capacity(n_centroids);
    let mut starts: Vec<usize> = Vec::with_capacity(digests.len());

    let mut count: f64 = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    let mut start: usize = 0;
    for digest in digests.iter() {
      starts.push(start);

      let curr_count: f64 = digest.count();
      if curr_count > 0.0 {
        min = min.min(digest.min);
        max = max.max(digest.max);
        count += curr_count;
        for centroid in &digest.centroids {
          centroids.push(centroid.clone());
          start += 1;
        }
      }
    }

    let mut digests_per_block: usize = 1;
    while digests_per_block < starts.len() {
      for i in (0..starts.len()).step_by(digests_per_block * 2) {
        if i + digests_per_block < starts.len() {
          let first = starts[i];
          let middle = starts[i + digests_per_block];
          let last = if i + 2 * digests_per_block < starts.len() {
            starts[i + 2 * digests_per_block]
          } else {
            centroids.len()
          };

          debug_assert!(first <= middle && middle <= last);
          Self::external_merge(&mut centroids, first, middle, last);
        }
      }

      digests_per_block *= 2;
    }

    let mut result = TDigest::new(max_size);
    let mut compressed: Vec<Centroid> = Vec::with_capacity(max_size);

    let mut k_limit: f64 = 1.0;
    let mut q_limit_times_count =
      Self::k_to_q(k_limit, max_size as f64) * (count);

    let mut iter_centroids = centroids.iter_mut();
    let mut curr = iter_centroids.next().unwrap();
    let mut weight_so_far = curr.weight();
    let mut sums_to_merge = 0_f64;
    let mut weights_to_merge = 0_f64;

    for centroid in iter_centroids {
      weight_so_far += centroid.weight();

      if weight_so_far <= q_limit_times_count {
        sums_to_merge += centroid.mean() * centroid.weight();
        weights_to_merge += centroid.weight();
      } else {
        result.sum += curr.add(sums_to_merge, weights_to_merge);
        sums_to_merge = 0_f64;
        weights_to_merge = 0_f64;
        compressed.push(curr.clone());
        q_limit_times_count = Self::k_to_q(k_limit, max_size as f64) * (count);
        k_limit += 1.0;
        curr = centroid;
      }
    }

    result.sum += curr.add(sums_to_merge, weights_to_merge);
    compressed.push(curr.clone());
    compressed.shrink_to_fit();
    compressed.sort();

    result.count = count;
    result.min = min;
    result.max = max;
    result.centroids = compressed;
    result
  }

  /// To estimate the value located at `q` quantile
  pub fn estimate_quantile(&self, q: f64) -> f64 {
    if self.centroids.is_empty() {
      return 0.0;
    }

    let count_ = self.count;
    let rank = q * count_;

    let mut pos: usize;
    let mut t;
    if q > 0.5 {
      if q >= 1.0 {
        return self.max();
      }

      pos = 0;
      t = count_;

      for (k, centroid) in self.centroids.iter().enumerate().rev() {
        t -= centroid.weight();

        if rank >= t {
          pos = k;
          break;
        }
      }
    } else {
      if q <= 0.0 {
        return self.min();
      }

      pos = self.centroids.len() - 1;
      t = 0_f64;

      for (k, centroid) in self.centroids.iter().enumerate() {
        if rank < t + centroid.weight() {
          pos = k;
          break;
        }

        t += centroid.weight();
      }
    }

    let mut delta = 0_f64;
    let mut min = self.min;
    let mut max = self.max;

    if self.centroids.len() > 1 {
      if pos == 0 {
        delta = self.centroids[pos + 1].mean() - self.centroids[pos].mean();
        max = self.centroids[pos + 1].mean();
      } else if pos == (self.centroids.len() - 1) {
        delta = self.centroids[pos].mean() - self.centroids[pos - 1].mean();
        min = self.centroids[pos - 1].mean();
      } else {
        delta = (self.centroids[pos + 1].mean()
          - self.centroids[pos - 1].mean())
          / 2.0;
        min = self.centroids[pos - 1].mean();
        max = self.centroids[pos + 1].mean();
      }
    }

    let value = self.centroids[pos].mean()
      + ((rank - t) / self.centroids[pos].weight() - 0.5) * delta;

    Self::clamp(value, min, max)
  }
}

#[cfg(debug_assertions)]
fn is_sorted(values: &[f64]) -> bool {
  values.windows(2).all(|w| w[0].total_cmp(&w[1]).is_le())
}

#[cfg(test)]
mod tests {
  use super::*;

  // A macro to assert the specified `quantile` estimated by `t` is within the
  // allowable relative error bound.
  macro_rules! assert_error_bounds {
    ($t:ident, quantile = $quantile:literal, want = $want:literal) => {
      assert_error_bounds!(
        $t,
        quantile = $quantile,
        want = $want,
        allowable_error = 0.01
      )
    };
    ($t:ident, quantile = $quantile:literal, want = $want:literal, allowable_error = $re:literal) => {
      let ans = $t.estimate_quantile($quantile);
      let expected: f64 = $want;
      let percentage: f64 = (expected - ans).abs() / expected;
      assert!(
        percentage < $re,
        "relative error {} is more than {}% (got quantile {}, want {})",
        percentage,
        $re,
        ans,
        expected
      );
    };
  }

  #[test]
  fn test_int64_uniform() {
    let values = (1i64..=1000).map(|v| v as f64).collect();

    let t = TDigest::new(100);
    let t = t.merge_unsorted_f64(values);

    assert_error_bounds!(t, quantile = 0.1, want = 100.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500.0);
    assert_error_bounds!(t, quantile = 0.9, want = 900.0);
  }

  #[test]
  fn test_centroid_addition_regression() {
    // https://github.com/MnO2/t-digest/pull/1

    let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let mut t = TDigest::new(10);

    for v in vals {
      t = t.merge_unsorted_f64(vec![v]);
    }

    assert_error_bounds!(t, quantile = 0.5, want = 1.0);
    assert_error_bounds!(t, quantile = 0.95, want = 2.0);
  }

  #[test]
  fn test_merge_unsorted_against_uniform_distro() {
    let t = TDigest::new(100);
    let values: Vec<_> = (1..=1_000_000).map(f64::from).collect();

    let t = t.merge_unsorted_f64(values);

    assert_error_bounds!(t, quantile = 1.0, want = 1_000_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990_000.0);
    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
  }

  #[test]
  fn test_merge_unsorted_against_skewed_distro() {
    let t = TDigest::new(100);
    let mut values: Vec<_> = (1..=600_000).map(f64::from).collect();
    values.resize(1_000_000, 1_000_000_f64);

    let t = t.merge_unsorted_f64(values);

    assert_error_bounds!(t, quantile = 0.99, want = 1_000_000.0);
    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
  }

  #[test]
  fn test_merge_digests() {
    let mut digests: Vec<TDigest> = Vec::new();

    for _ in 1..=100 {
      let t = TDigest::new(100);
      let values: Vec<_> = (1..=1_000).map(f64::from).collect();
      let t = t.merge_unsorted_f64(values);
      digests.push(t)
    }

    let t = TDigest::merge_digests(&digests);

    assert_error_bounds!(t, quantile = 1.0, want = 1000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990.0);
    assert_error_bounds!(
      t,
      quantile = 0.01,
      want = 10.0,
      allowable_error = 0.2
    );
    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500.0);
  }

  #[test]
  fn test_size() {
    let t = TDigest::new(10);
    let t = t.merge_unsorted_f64(vec![0.0, 1.0]);

    assert_eq!(t.size(), 96);
  }
}
