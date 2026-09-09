// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow::compute::{max, min};
use arrow::datatypes::{DataType, Schema};
use datafusion_common::{
    DataFusionError, Result, ScalarValue, downcast_value, internal_err, plan_err,
};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

/// Evaluates a physical expression to extract its scalar value.
///
/// This is used to extract constant values from expressions (like percentile parameters)
/// by evaluating them against an empty record batch.
pub(crate) fn get_scalar_value(expr: &Arc<dyn PhysicalExpr>) -> Result<ScalarValue> {
    let empty_schema = Arc::new(Schema::empty());
    let batch = RecordBatch::new_empty(Arc::clone(&empty_schema));
    if let ColumnarValue::Scalar(s) = expr.evaluate(&batch)? {
        Ok(s)
    } else {
        internal_err!("Didn't expect ColumnarValue::Array")
    }
}

/// Validates that a percentile scalar is a Float32/Float64 value between 0.0 and 1.0.
fn scalar_to_percentile(scalar_value: ScalarValue, fn_name: &str) -> Result<f64> {
    let percentile = match scalar_value {
        ScalarValue::Float32(Some(value)) => value as f64,
        ScalarValue::Float64(Some(value)) => value,
        ScalarValue::Float32(None) | ScalarValue::Float64(None) => {
            return plan_err!(
                "Percentile value for '{fn_name}' must be Float32 or Float64 (got null)"
            );
        }
        sv => {
            return plan_err!(
                "Percentile value for '{fn_name}' must be Float32 or Float64 (got data type {})",
                sv.data_type()
            );
        }
    };

    // Ensure the percentile is between 0 and 1.
    if !(0.0..=1.0).contains(&percentile) {
        return plan_err!(
            "Percentile value must be between 0.0 and 1.0 inclusive, {percentile} is invalid"
        );
    }
    Ok(percentile)
}

/// A percentile argument that may already be known at accumulator-construction
/// time (a literal, or an expression that is foldable without any input row),
/// or may only be resolvable once real input data is available - e.g. a
/// projected column that happens to be constant for every row of the
/// aggregation (`SELECT approx_percentile_cont(y, m) FROM (SELECT x+1 AS y,
/// 0.5 AS m FROM ...)`).
///
/// Deferred resolution reads the percentile out of the array that the
/// aggregation engine already evaluates and passes to
/// `Accumulator::update_batch` for this argument - no new column-constant
/// analysis is needed. Every call to `resolve` also checks that the batch's
/// min/max agree (and, once resolved, that later batches agree with the
/// cached value), so a genuinely non-constant argument produces a clear error
/// instead of silently using an arbitrary row's value.
#[derive(Debug, Clone)]
pub(crate) enum PercentileParam {
    Resolved(f64),
    Pending { fn_name: String },
}

impl PercentileParam {
    /// Try to resolve the percentile eagerly (today's behavior, for literals
    /// and constant-foldable expressions). If the expression can't be
    /// evaluated without row data (i.e. it references a column), defer
    /// resolution to the first batch instead of erroring here.
    pub(crate) fn try_new(expr: &Arc<dyn PhysicalExpr>, fn_name: &str) -> Result<Self> {
        match get_scalar_value(expr) {
            Ok(scalar_value) => {
                Ok(Self::Resolved(scalar_to_percentile(scalar_value, fn_name)?))
            }
            Err(_) => Ok(Self::Pending {
                fn_name: fn_name.to_string(),
            }),
        }
    }

    /// Resolve (if not already resolved) using the array evaluated for this
    /// argument on the current batch, and validate that the argument is
    /// indeed constant across the batch (and, if already resolved, that it
    /// agrees with the previously resolved value).
    ///
    /// A batch with no informative (non-null) values is a no-op, not an
    /// error: with multiple partitions/batches feeding the same accumulator
    /// (or group), any individual batch may legitimately be empty or
    /// all-null while a later one still carries the real, constant value.
    /// Whether the percentile was *ever* successfully resolved is checked
    /// separately, once, by `get()` at evaluate time.
    ///
    /// The percentile argument's signature is normally coerced to `Float64`
    /// by the UDAFs, but callers that build the physical expression directly
    /// (bypassing logical-plan coercion, e.g. some unit tests) may still
    /// pass a `Float32` array - so both are handled here, matching
    /// `scalar_to_percentile`'s handling of the eager (literal) path.
    pub(crate) fn resolve(&mut self, array: &ArrayRef) -> Result<()> {
        if array.null_count() >= array.len() {
            return Ok(());
        }

        let fn_name = self.fn_name().to_string();
        let (batch_min, batch_max) = match array.data_type() {
            DataType::Float64 => {
                let float_array = downcast_value!(array, Float64Array);
                (min(float_array), max(float_array))
            }
            DataType::Float32 => {
                let float_array = downcast_value!(array, Float32Array);
                (
                    min(float_array).map(|v| v as f64),
                    max(float_array).map(|v| v as f64),
                )
            }
            data_type => {
                return plan_err!(
                    "Percentile value for '{fn_name}' must be Float32 or Float64 (got {data_type})"
                );
            }
        };
        let batch_min = batch_min.ok_or_else(|| {
            DataFusionError::Internal("expected a non-null percentile value".to_string())
        })?;
        let batch_max = batch_max.ok_or_else(|| {
            DataFusionError::Internal("expected a non-null percentile value".to_string())
        })?;
        if batch_min != batch_max {
            return plan_err!(
                "Percentile value for '{fn_name}' must be constant across the aggregation, found differing values"
            );
        }

        match self {
            Self::Resolved(resolved) => {
                if batch_min != *resolved {
                    return plan_err!(
                        "Percentile value for '{fn_name}' must be constant across the aggregation, found differing values"
                    );
                }
            }
            Self::Pending { .. } => {
                let resolved = scalar_to_percentile(
                    ScalarValue::Float64(Some(batch_min)),
                    &fn_name,
                )?;
                *self = Self::Resolved(resolved);
            }
        }

        Ok(())
    }

    fn fn_name(&self) -> &str {
        match self {
            Self::Resolved(_) => "<resolved>",
            Self::Pending { fn_name } => fn_name,
        }
    }

    /// Returns the resolved percentile, or an error if `resolve` has never
    /// successfully been called (i.e. no batch with a non-null percentile
    /// value has been seen yet).
    pub(crate) fn get(&self) -> Result<f64> {
        match self {
            Self::Resolved(value) => Ok(*value),
            Self::Pending { fn_name } => {
                plan_err!(
                    "Percentile value for '{fn_name}' could not be determined: no non-null percentile value was seen"
                )
            }
        }
    }
}
