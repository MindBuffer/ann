//!
//!  neuron.rs
//!
//!  Created by Mitchell Nordine at 12:18AM on October 20, 2014.
//!
//!

use utils::logistic;
use utils::map_range;

/// The Weight will always be represented as a value between 0.0 - 1.0.
pub type Weight = f32;

/// Represents a connection to some form of neuronal input.
pub struct Synapse<I>(I, Weight);

impl<I> Synapse<I> {
    fn set_input(&mut self, input: I) { let Synapse(ref mut i, _) = *self; *i = input }
    fn set_weight(&mut self, weight: f32) { let Synapse(_, ref mut w) = *self; *w = weight }
}

/// A trait to be implemented for all neuron types.
pub trait Neuron<I: Clone, O> {
    fn evaluate(&self) -> O;
    fn get_synapses(&self) -> &Vec<Synapse<I>>;
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<I>>;
    fn set_inputs(&mut self, new_inputs: &Vec<I>) {
        assert!(new_inputs.len() == self.get_synapses().len(),
                "The number of inputs ({}) must match the number of synapses ({}).",
                new_inputs.len(), self.get_synapses().len());
        for (i, synapse) in self.get_synapses_mut().iter_mut().enumerate() {
            synapse.set_input((*new_inputs)[i].clone());
        }
    }
}

/// A type of neuron that linearly evaluates it's inputs and returns
/// the real-value result.
pub struct Linear {
    synapses: Vec<Synapse<f32>>,
}

impl Neuron<f32, f32> for Linear {
    fn get_synapses(&self) -> &Vec<Synapse<f32>> { &self.synapses }
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<f32>> { &mut self.synapses }
    fn evaluate(&self) -> f32 {
        let total_input = self.synapses.iter().fold(0.0, |total, &Synapse(input, weight)| {
            total + input * weight
        });
        total_input / self.synapses.len() as f32
    }
}

/// Similar to the linear neuron, however rather than returning the
/// real_valued output, it will compare the value to a threshold,
/// returning true if greater than the threshold and false if less.
pub struct BinaryThreshold {
    synapses: Vec<Synapse<f32>>,
    /// A threshold that be between 0.0 and 1.0.
    threshold: f32,
}

impl Neuron<f32, bool> for BinaryThreshold {
    fn get_synapses(&self) -> &Vec<Synapse<f32>> { &self.synapses }
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<f32>> { &mut self.synapses }
    fn evaluate(&self) -> bool {
        let total_input = self.synapses.iter().fold(0.0, |total, &Synapse(input, weight)| {
            total + input * weight
        });
        let normalised = total_input / self.synapses.len() as f32;
        if normalised > self.threshold { true } else { false }
    }
}

/// This neuron, similarly to the Linear neuron, computes a linear
/// weighted sum of it's inputs, however it returns a non-linear
/// result dependent upon a threshold. Below the threshold None is
/// returned, however above the threshold the real-value is returned.
pub struct RectifiedLinear {
    synapses: Vec<Synapse<f32>>,
    /// A threshold that be between 0.0 and 1.0.
    threshold: f32,
}
pub type LinearThreshold = RectifiedLinear;

impl Neuron<f32, Option<f32>> for RectifiedLinear {
    fn get_synapses(&self) -> &Vec<Synapse<f32>> { &self.synapses }
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<f32>> { &mut self.synapses }
    fn evaluate(&self) -> Option<f32> {
        let total_input = self.synapses.iter().fold(0.0, |total, &Synapse(input, weight)| {
            total + input * weight
        });
        let normalised = total_input / self.synapses.len() as f32;
        if normalised > self.threshold {
            Some(map_range(normalised, self.threshold, 1.0, 0f32, 1.0))
        } else { None }
    }
}

/// A type of neuron that gives a real-valued output that is a smooth
/// and bounded function of their total input. Rather than linearly
/// evaluating the weighted inputs, it applies the logistic function
/// which returns nice, smooth derivatives.
pub struct Sigmoid {
    synapses: Vec<Synapse<f32>>,
}
pub type Logistic = Sigmoid;

impl Neuron<f32, f32> for Sigmoid {
    fn get_synapses(&self) -> &Vec<Synapse<f32>> { &self.synapses }
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<f32>> { &mut self.synapses }
    fn evaluate(&self) -> f32 {
        let total_input = self.synapses.iter().fold(0.0, |total, &Synapse(input, weight)| {
            total + input * weight
        });
        // Map the result so that the logistic function result will be 0.0 - 1.0.
        let mapped = map_range(total_input, 0.0, self.synapses.len() as f32, -10f32, 10.0);
        logistic(mapped)
    }
}

/// Similar to the Sigmoid neuron, however rather than giving a
/// real-valued output, they return a true or false based on their
/// resulting probability.
pub struct StochasticBinary {
    synapses: Vec<Synapse<f32>>,
}

impl Neuron<f32, bool> for StochasticBinary {
    fn get_synapses(&self) -> &Vec<Synapse<f32>> { &self.synapses }
    fn get_synapses_mut(&mut self) -> &mut Vec<Synapse<f32>> { &mut self.synapses }
    fn evaluate(&self) -> bool {
        let total_input = self.synapses.iter().fold(0.0, |total, &Synapse(input, weight)| {
            total + input * weight
        });
        // Map the result so that the logistic function result will be 0.0 - 1.0.
        let mapped = map_range(total_input, 0.0, self.synapses.len() as f32, -10f32, 10.0);
        if logistic(mapped) > 0.5 { true } else { false }
    }
}

