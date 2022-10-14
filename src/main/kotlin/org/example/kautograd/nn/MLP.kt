package org.example.kautograd.nn

import org.example.kautograd.engine.Value

class MLP(nInputs: Int, vararg nOutputs: Int) {

    private val layers: List<Layer>

    init {
        val sizes: Array<Int> = arrayOf(nInputs).plus(nOutputs.toTypedArray())
        layers = IntRange(0, nOutputs.size - 1).map { i -> Layer(sizes[i], sizes[i + 1]) }
    }

    fun call(x: List<Value>): List<Value> {
        var layerOutput = x;
        // Call each layer with output from previous layer
        layers.forEach {
            layerOutput = it.call(layerOutput)
        }
        return layerOutput
    }

    fun getParameters(): List<Value> {
        return layers.map { l -> l.getParameters() }.flatten()
    }


}
