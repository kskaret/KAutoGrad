package org.example.kautograd.nn

import org.example.kautograd.engine.Value

class Layer(private val nInputs: Int, nOutputs: Int) {

    fun call(x: List<Value>): List<Value> {
        return neurons.map { n -> n.call(x) }.toList();
    }

    fun getParameters(): List<Value> {
        return neurons.map { n -> n.getParameters() }.flatten()
    }

    private val neurons: List<Neuron> = List(nOutputs) { Neuron(nInputs) }
}