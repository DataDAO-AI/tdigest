use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

const DEFAULT_MAX_SIZE: usize = 100;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct WeightedValue {
  value: f64,
  weight: f64,
}

impl Eq for WeightedValue {}

impl PartialOrd for WeightedValue {
  fn partial_cmp(&self, other: &WeightedValue) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl From<(f64, f64)> for WeightedValue {
  fn from((value, weight): (f64, f64)) -> Self {
    Self { value, weight }
  }
}

impl Ord for WeightedValue {
  fn cmp(&self, other: &WeightedValue) -> Ordering {
    self.value.total_cmp(&other.value)
  }
}

impl Default for WeightedValue {
  fn default() -> Self {
    WeightedValue {
      value: 0_f64,
      weight: 1_f64,
    }
  }
}

impl WeightedValue {
  pub fn new(value: f64, weight: f64) -> Self {
    WeightedValue { value, weight }
  }

  fn add_sum(&mut self, sum: f64, weight: f64) -> f64 {
    let new_sum = sum + self.sum();
    self.weight += weight;
    self.value = new_sum / self.weight;
    new_sum
  }

  pub fn value(&self) -> f64 {
    self.value
  }

  fn sum(&self) -> f64 {
    self.value * self.weight
  }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TDigest {
  centroids: Vec<WeightedValue>,
  pub max_size: usize,
  pub sum: f64,
  pub weight_sum: f64,
  pub max: f64,
  pub min: f64,
}

impl TDigest {
  pub fn new_with_max_size(max_size: usize) -> Self {
    TDigest {
      centroids: Vec::new(),
      max_size,
      sum: 0_f64,
      weight_sum: 0_f64,
      max: f64::NAN,
      min: f64::NAN,
    }
  }

  pub fn new() -> Self {
    TDigest::new_with_max_size(DEFAULT_MAX_SIZE)
  }

  pub fn compress_from_weighted_values(
    mut values: Vec<WeightedValue>,
    max_size: usize,
    weight_sum: f64,
    min: f64,
    max: f64,
  ) -> TDigest {
    let mut result = TDigest::new_with_max_size(max_size);

    let mut iter_values = values.iter_mut();
    match iter_values.next() {
      None => result,
      Some(first_value) => {
        let mut compressed: Vec<WeightedValue> = Vec::with_capacity(max_size);
        let mut weight_scaled_q_limit =
          Self::k_to_q(1.0, max_size as f64) * weight_sum;
        let mut curr = first_value;
        let mut weight_so_far = curr.weight;
        let mut sums_to_merge = 0_f64;
        let mut weights_to_merge = 0_f64;

        let mut k_limit = 1_f64;
        for centroid in iter_values {
          weight_so_far += centroid.weight;

          if weight_so_far <= weight_scaled_q_limit {
            sums_to_merge += centroid.sum();
            weights_to_merge += centroid.weight;
          } else {
            result.sum += curr.add_sum(sums_to_merge, weights_to_merge);
            sums_to_merge = 0_f64;
            weights_to_merge = 0_f64;
            compressed.push(curr.clone());
            weight_scaled_q_limit =
              Self::k_to_q(k_limit, max_size as f64) * weight_sum;
            k_limit += 1.0;
            curr = centroid;
          }
        }

        result.sum += curr.add_sum(sums_to_merge, weights_to_merge);
        compressed.push(curr.clone());
        compressed.shrink_to_fit();
        compressed.sort();

        result.weight_sum = weight_sum;
        result.min = min;
        result.max = max;
        result.centroids = compressed;
        result
      }
    }
  }

  /// Size in bytes including `Self`.
  pub fn size(&self) -> usize {
    std::mem::size_of_val(self)
      + (std::mem::size_of::<WeightedValue>() * self.centroids.capacity())
  }
}

impl Default for TDigest {
  fn default() -> Self {
    Self::new()
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

  pub fn merge_weighted_values(
    self,
    mut values: Vec<WeightedValue>,
  ) -> TDigest {
    let (weight_sum, min, max) = values.iter().fold(
      (self.weight_sum, self.min, self.max),
      |(weight_sum, min, max), weighted_value| {
        (
          weight_sum + weighted_value.weight,
          min.min(weighted_value.value),
          max.max(weighted_value.value),
        )
      },
    );

    let original_centroid_count = self.centroids.len();
    let mut all_values = self.centroids;
    all_values.append(&mut values);
    let value_count = all_values.len();

    Self::external_merge(
      &mut all_values,
      0,
      original_centroid_count,
      value_count,
    );

    Self::compress_from_weighted_values(
      all_values,
      self.max_size,
      weight_sum,
      min,
      max,
    )
  }

  pub fn merge_values(self, mut values: Vec<f64>) -> TDigest {
    values.sort_by(|a, b| a.total_cmp(b));
    self.merge_sorted_values(&values)
  }

  pub fn merge_sorted_values(self, sorted_values: &[f64]) -> TDigest {
    #[cfg(debug_assertions)]
    debug_assert!(
      is_sorted(sorted_values),
      "unsorted input to TDigest::merge_sorted_values"
    );

    if sorted_values.is_empty() {
      return self.clone();
    }

    let mut result = TDigest::new_with_max_size(self.max_size);
    result.weight_sum = self.weight_sum + (sorted_values.len() as f64);

    let maybe_min = *sorted_values.first().unwrap();
    let maybe_max = *sorted_values.last().unwrap();

    if self.weight_sum > 0.0 {
      result.min = self.min.min(maybe_min);
      result.max = self.max.max(maybe_max);
    } else {
      result.min = maybe_min;
      result.max = maybe_max;
    }

    let mut compressed: Vec<WeightedValue> = Vec::with_capacity(self.max_size);

    let mut weight_scaled_q_limit =
      Self::k_to_q(1.0, self.max_size as f64) * result.weight_sum;

    let mut iter_centroids = self.centroids.iter().peekable();
    let mut iter_sorted_values = sorted_values.iter().peekable();

    let mut curr: WeightedValue = if let Some(c) = iter_centroids.peek() {
      let curr = **iter_sorted_values.peek().unwrap();
      if c.value < curr {
        iter_centroids.next().unwrap().clone()
      } else {
        WeightedValue::new(*iter_sorted_values.next().unwrap(), 1.0)
      }
    } else {
      WeightedValue::new(*iter_sorted_values.next().unwrap(), 1.0)
    };

    let mut weight_so_far = curr.weight;

    let mut sums_to_merge = 0_f64;
    let mut weights_to_merge = 0_f64;

    let mut k_limit = 2_f64;
    while iter_centroids.peek().is_some() || iter_sorted_values.peek().is_some()
    {
      let next: WeightedValue = if let Some(c) = iter_centroids.peek() {
        if iter_sorted_values.peek().is_none()
          || c.value < **iter_sorted_values.peek().unwrap()
        {
          iter_centroids.next().unwrap().clone()
        } else {
          WeightedValue::new(*iter_sorted_values.next().unwrap(), 1.0)
        }
      } else {
        WeightedValue::new(*iter_sorted_values.next().unwrap(), 1.0)
      };

      let next_sum = next.sum();
      weight_so_far += next.weight;

      if weight_so_far <= weight_scaled_q_limit {
        sums_to_merge += next_sum;
        weights_to_merge += next.weight;
      } else {
        result.sum += curr.add_sum(sums_to_merge, weights_to_merge);
        sums_to_merge = 0_f64;
        weights_to_merge = 0_f64;

        compressed.push(curr.clone());
        weight_scaled_q_limit =
          Self::k_to_q(k_limit, self.max_size as f64) * result.weight_sum;
        k_limit += 1.0;
        curr = next;
      }
    }

    result.sum += curr.add_sum(sums_to_merge, weights_to_merge);
    compressed.push(curr);
    compressed.shrink_to_fit();
    compressed.sort();

    result.centroids = compressed;
    result
  }

  fn external_merge(
    centroids: &mut [WeightedValue],
    first: usize,
    middle: usize,
    last: usize,
  ) {
    let mut result: Vec<WeightedValue> = Vec::with_capacity(centroids.len());

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
    let mut centroids: Vec<WeightedValue> = Vec::with_capacity(n_centroids);
    let mut starts: Vec<usize> = Vec::with_capacity(digests.len());

    let mut weight_sum: f64 = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    let mut start: usize = 0;
    for digest in digests.iter() {
      starts.push(start);

      let curr_weight_sum: f64 = digest.weight_sum;
      if curr_weight_sum > 0.0 {
        min = min.min(digest.min);
        max = max.max(digest.max);
        weight_sum += curr_weight_sum;
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

    Self::compress_from_weighted_values(
      centroids, max_size, weight_sum, min, max,
    )
  }

  /// To estimate the value located at `q` quantile
  pub fn estimate_quantile(&self, q: f64) -> f64 {
    if self.centroids.is_empty() {
      return 0.0;
    }

    let rank = q * self.weight_sum;

    let mut pos: usize;
    let mut t;
    if q > 0.5 {
      if q >= 1.0 {
        return self.max;
      }

      pos = 0;
      t = self.weight_sum;

      for (k, centroid) in self.centroids.iter().enumerate().rev() {
        t -= centroid.weight;

        if rank >= t {
          pos = k;
          break;
        }
      }
    } else {
      if q <= 0.0 {
        return self.min;
      }

      pos = self.centroids.len() - 1;
      t = 0_f64;

      for (k, centroid) in self.centroids.iter().enumerate() {
        if rank < t + centroid.weight {
          pos = k;
          break;
        }

        t += centroid.weight;
      }
    }

    let mut delta = 0_f64;
    let mut min = self.min;
    let mut max = self.max;

    if self.centroids.len() > 1 {
      if pos == 0 {
        delta = self.centroids[pos + 1].value - self.centroids[pos].value;
        max = self.centroids[pos + 1].value;
      } else if pos == (self.centroids.len() - 1) {
        delta = self.centroids[pos].value - self.centroids[pos - 1].value;
        min = self.centroids[pos - 1].value;
      } else {
        delta =
          (self.centroids[pos + 1].value - self.centroids[pos - 1].value) / 2.0;
        min = self.centroids[pos - 1].value;
        max = self.centroids[pos + 1].value;
      }
    }

    let value = self.centroids[pos].value
      + ((rank - t) / self.centroids[pos].weight - 0.5) * delta;

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

    let t = TDigest::new();
    let t = t.merge_values(values);

    assert_error_bounds!(t, quantile = 0.1, want = 100.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500.0);
    assert_error_bounds!(t, quantile = 0.9, want = 900.0);
  }

  #[test]
  fn test_centroid_addition_regression() {
    // https://github.com/MnO2/t-digest/pull/1

    let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let mut t = TDigest::new_with_max_size(10);

    for v in vals {
      t = t.merge_values(vec![v]);
    }

    assert_error_bounds!(t, quantile = 0.5, want = 1.0);
    assert_error_bounds!(t, quantile = 0.95, want = 2.0);
  }

  #[test]
  fn test_merge_values_against_uniform_distribution() {
    let t = TDigest::new();
    let values: Vec<f64> = (1..=1_000_000).map(f64::from).collect();

    let t = t.merge_values(values);

    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990_000.0);
    assert_error_bounds!(t, quantile = 1.0, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_values_against_skewed_distribution() {
    let t = TDigest::new();
    let first_segment = (1..=600_000).map(f64::from);
    let second_segment = std::iter::repeat(1_000_000_f64).take(400_000);
    let values: Vec<f64> = first_segment.chain(second_segment).collect();

    let t = t.merge_values(values);

    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_weighted_values_against_uniform_distribution() {
    let t = TDigest::new();
    let values: Vec<WeightedValue> =
      (1..=1_000_000).map(|i| (i as f64, 1f64).into()).collect();

    let t = t.merge_weighted_values(values);

    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990_000.0);
    assert_error_bounds!(t, quantile = 1.0, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_weighted_values_against_weighted_uniform_distribution() {
    let t = TDigest::new();
    let first_segment = (1..=500_000).map(|i| (i as f64, 1f64).into());
    let second_segment =
      (1..=250_000).map(|i| ((500_000 + i * 2) as f64, 2f64).into());
    let values: Vec<WeightedValue> =
      first_segment.chain(second_segment).collect();

    let t = t.merge_weighted_values(values);

    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.25, want = 250_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.75, want = 750_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990_000.0);
    assert_error_bounds!(t, quantile = 1.0, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_weighted_values_against_skewed_distribution() {
    let t = TDigest::new();
    let first_segment = (1..=600_000).map(|i| (i as f64, 1f64).into());
    let second_segment =
      std::iter::repeat((1_000_000_f64, 1f64).into()).take(400_000);
    let values: Vec<WeightedValue> =
      first_segment.chain(second_segment).collect();

    let t = t.merge_weighted_values(values);

    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_weighted_values_against_weighted_skewed_distribution() {
    let t = TDigest::new_with_max_size(1000);
    let first_segment = (1..=300_000).map(|i| (i as f64, 1f64).into());
    let second_segment =
      (1..=150_000).map(|i| ((300_000 + i * 2) as f64, 2f64).into());
    let third_segment =
      std::iter::repeat((1_000_000_f64, 1f64).into()).take(400_000);
    let values: Vec<WeightedValue> = first_segment
      .chain(second_segment)
      .chain(third_segment)
      .collect();

    let t = t.merge_weighted_values(values);

    assert_error_bounds!(t, quantile = 0.01, want = 10_000.0);
    assert_error_bounds!(t, quantile = 0.5, want = 500_000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 1_000_000.0);
  }

  #[test]
  fn test_merge_identical_digests() {
    let mut digests: Vec<TDigest> = Vec::new();

    for _ in 1..=100 {
      let t = TDigest::new();
      let values: Vec<f64> = (1..=1_000).map(f64::from).collect();
      let t = t.merge_values(values);
      digests.push(t)
    }

    let t = TDigest::merge_digests(&digests);

    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(
      t,
      quantile = 0.01,
      want = 10.0,
      allowable_error = 0.2
    );
    assert_error_bounds!(t, quantile = 0.5, want = 500.0);
    assert_error_bounds!(t, quantile = 0.99, want = 990.0);
    assert_error_bounds!(t, quantile = 1.0, want = 1000.0);
  }

  #[test]
  fn test_merge_distinct_digests() {
    let mut digests: Vec<TDigest> = Vec::new();

    for i in 0..100 {
      let t = TDigest::new();
      let values: Vec<f64> =
        (1..=1_000).map(|x| f64::from(x + i * 1000)).collect();
      let t = t.merge_values(values);
      digests.push(t)
    }

    let t = TDigest::merge_digests(&digests);

    assert_error_bounds!(t, quantile = 0.0, want = 1.0);
    assert_error_bounds!(
      t,
      quantile = 0.01,
      want = 1000.0,
      allowable_error = 0.2
    );
    assert_error_bounds!(t, quantile = 0.5, want = 50000.0);
    assert_error_bounds!(t, quantile = 0.99, want = 99000.0);
    assert_error_bounds!(t, quantile = 1.0, want = 100000.0);
  }

  #[test]
  fn test_size() {
    let t = TDigest::new_with_max_size(10);
    let t = t.merge_values(vec![0.0, 1.0]);

    assert_eq!(t.size(), 96);
  }
}
