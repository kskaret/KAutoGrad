package org.example.kautograd.nn

import org.example.kautograd.engine.Value


class Neuron(nInputs: Int) {
    fun call(x: List<Value>): Value {
        var sum: Value = b

        for (i in weights.indices) {
            val wx: Value = weights[i] * x[i]
            sum += wx
        }

        return Value.tanh(sum)
    }

    fun getParameters(): List<Value> {
        return weights
    }

    private val weights: List<Value> = List(nInputs) { Value.random() }
    private val b = Value.random()
}