//!  
//!  Example from "Neural Networks Demystified", by Stephen Welch.
//!
//!
//!  INPUTS        || OUTPUT
//!  ============= || ======
//!  Sleep | Study || Score
//!  ----- | ----- || -----
//!  3     | 5     || 75
//!  5     | 1     || 82
//!  10    | 2     || 93
//!

extern crate ann;


fn main() {

    use ann::matrix::{self, Mat};
    use ann::network::Network;


    // Data Constants.
    const MAX_SCORE: f32 = 100.0;
    const NUM_EXAMPLES: usize = 3;

    // Neural Network Hyperparameters.
    const NUM_INPUTS: usize = 2;
    const NUM_HIDDEN: usize = 3;
    const NUM_OUTPUTS: usize = 1;


    // Construct our input data matrix.
    let mut input_data = Mat::from_col_vec(NUM_EXAMPLES, NUM_INPUTS, &[
        3.0, 5.0, 10.0,     // Sleep
        5.0, 1.0, 2.0,      // Study
    ]);

    // Construct our matrix of the corresponding correct output.
    let mut correct_output = Mat::from_col_vec(NUM_EXAMPLES, NUM_OUTPUTS, &[
        75.0, 82.0, 93.0,   // Score
    ]);

    print!("input_data: \n{:?}", &input_data);
    print!("correct_output: \n{:?}", &correct_output);


    // Normalise input data.
    matrix::normalise_cols(&mut input_data);

    // Normalise correct output.
    matrix::update_elems(&mut correct_output, |score| score / MAX_SCORE);

    print!("Normalised input_data:\n{:?}", &input_data);
    print!("Normalised correct_output:\n{:?}", &correct_output);


    // Construct network.
    let net = Network::new(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS);


    // Calculate scores with feed forward data.
    let guesses = net.forward(input_data);

    println!("Guesses:\n{:?}", &guesses);

}


