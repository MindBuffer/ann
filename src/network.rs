
use matrix::{self, Mat};


/// An Artificial Neural Network with a single hidden layer.
/// The `Mat`rices are column-major `Vec`tors.
#[derive(Clone, Debug)]
pub struct Network {
    /// The Matrix for our hidden weights.
    /// - One column for each hidden neuron.
    /// - One row for each input.
    hidden_weights: Mat<f32>,
    /// The Matrix for our output weights.
    /// - One column for each output neuron.
    /// - One row for each hidden neuron.
    output_weights: Mat<f32>,
}


impl Network {

    /// Construct an Artificial Neural Network.
    /// Initialise all of the weights as random values.
    pub fn new(n_inputs: usize, n_hidden: usize, n_outputs: usize) -> Network {
        Network {
            hidden_weights: Mat::from_fn(n_inputs, n_hidden,  |_, _| ::rand::random()),
            output_weights: Mat::from_fn(n_hidden, n_outputs, |_, _| ::rand::random()),
        }
    }

    /// Generate a matrix of guesses by feeding forward some matrix of inputs.
    /// The `Mat`rices are column-major `Vec`tors.
    /// For the input data, each column is a different input and each row is a different example.
    pub fn forward(&self, input_data: Mat<f32>) -> Mat<f32> {
        let Network { ref hidden_weights, ref output_weights } = *self;

        // Determine the hidden layer's activity matrix by multiplying the input matrix by our
        // matrix of hidden layer weights.
        let mut hidden_activity = input_data * hidden_weights.clone();

        // Apply our activation function to every element in our hidden_activity matrix.
        matrix::update_elems(&mut hidden_activity, logistic);

        // Determine the output activity by multiplying our hidden_activity with the output
        // weights.
        let mut output_activity = hidden_activity * output_weights.clone();

        // Apply our activation function to every element in our output_activity matrix.
        matrix::update_elems(&mut output_activity, logistic);

        output_activity
    }

}


/// The sum of the error of each guess compared to its correct result.
pub fn cost(guesses: &[f32], correct: &[f32]) -> f32 {
    guesses.iter().zip(correct.iter())
        .fold(0.0, |so_far, (guess, correct)| {
            let error = correct - guess;
            so_far + error.powf(2.0) / 2.0
        })
}


/// The logistic function.
pub fn logistic(z: f32) -> f32 {
    use std::f32::consts::E;
    1.0 / (1.0 + E.powf(-z))
}

