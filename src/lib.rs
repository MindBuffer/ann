
/// An experimental implementation of an Artificial Neural Network.
/// Hopefully, this will evolve more and more over time as I learn
/// more about ANNs and machine learning.

pub use neuron::Linear as LinearNeuron;
pub use neuron::BinaryThreshold as BinaryThresholdNeuron;
pub use neuron::RectifiedLinear as RectifiedLinearNeuron;
pub use neuron::LinearThreshold as LinearThresholdNeuron;
pub use neuron::Sigmoid as SigmoidNeuron;
pub use neuron::Logistic as LogisticNeuron;
pub use neuron::StochasticBinary as StochasticBinaryNeuron;

mod neuron;
mod utils;

