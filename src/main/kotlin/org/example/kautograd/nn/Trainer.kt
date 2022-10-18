package org.example.kautograd.nn

import org.example.kautograd.engine.Value

class Trainer {

    fun train(nn: MLP, xs: List<List<Value>>, yTargets: List<Value>) {
        
        // training loop
        for (k in 0..19) {

            // forward pass
            val yPredictions: List<Value> = xs.map { x -> nn.call(x)[0] }

            // loss
            val loss: Value = lossFunction(yPredictions, yTargets)

            // backward pass
            for (p in nn.getParameters()) {
                p.resetGrad()
            }
            loss.backwardsPropagation()

            // update
            val parameters: List<Value> = nn.getParameters()
            for (p in parameters) {
                p.adjustWithGradient(0.05)
            }
            println("Step $k, loss: ${loss.data}")
        }
    }

    private fun lossFunction(yPredictions: List<Value>, yTargets: List<Value>): Value {
        var loss = Value(0.0)

        for (i in yTargets.indices) {
            val sub: Value = yTargets[i] - yPredictions[i]
            loss += sub * sub
        }

        return loss
    }

}
