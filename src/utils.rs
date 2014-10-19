//!
//!  utils.rs
//!
//!  Created by Mitchell Nordine at 11:03PM on October 19, 2014.
//!
//!

/// Map a value from the input range to the output range.
pub fn map_range<X: Num + Copy + FromPrimitive + ToPrimitive,
                 Y: Num + Copy + FromPrimitive + ToPrimitive>
(val: X, in_min: X, in_max: X, out_min: Y, out_max: Y) -> Y {
    let (val_f, in_min_f, in_max_f, out_min_f, out_max_f) = (
        val.to_f64().unwrap(),
        in_min.to_f64().unwrap(),
        in_max.to_f64().unwrap(),
        out_min.to_f64().unwrap(),
        out_max.to_f64().unwrap(),
    );
    if (in_min_f - in_max_f).abs() < Float::epsilon() {
        println!("jmath Warning: map(): avoiding possible divide by zero, \
                 in_min ({}) and in_max({})", in_min_f, in_max_f);
        return out_min;
    }
    FromPrimitive::from_f64(
        (val_f - in_min_f) / (in_max_f - in_min_f) * (out_max_f - out_min_f) + out_min_f
    ).unwrap()
}

/// The logistic function.
pub fn logistic(z: f32) -> f32 {
    let e: f32 = Float::e();
    1.0 / (1.0 + e.powf(-z))
}

