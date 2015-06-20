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

    use ann::matrix::Mat;
    use ann::network::Network;


    // Hyperparameters.
    const MAX_SLEEP: f32 = 10.0;
    const MAX_STUDY: f32 = 10.0;
    const MAX_SCORE: f32 = 100.0;
    const NUM_EXAMPLES: usize = 3;
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
    {
        let mut inputs = input_data.as_mut_vec().chunks_mut(NUM_EXAMPLES);
        for sleep in inputs.next().unwrap() {
            *sleep = *sleep / MAX_SLEEP;
        }
        for study in inputs.next().unwrap() {
            *study = *study / MAX_STUDY;
        }
    }

    // Normalise correct output.
    for score in correct_output.as_mut_vec().iter_mut() {
        *score = *score / MAX_SCORE;
    }

    print!("Normalised input_data:\n{:?}", &input_data);
    print!("Normalised correct_output:\n{:?}", &correct_output);


    // Construct network.
    let net = Network::new(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS);


    // Calculate scores with feed forward data.
    let guesses = net.forward(input_data);

    println!("Guesses:\n{:?}", &guesses);

}


