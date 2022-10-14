package org.example.kautograd

import org.example.kautograd.engine.Value
import org.example.kautograd.nn.MLP
import org.example.kautograd.nn.Trainer
import org.junit.Assert
import org.junit.Test

class SimpleExampleTest {

    @Test
    fun simpleExample() {

        val nn = MLP(3, 4, 4, 1)

        val xs: List<List<Value>> = listOf(
            Value.listOf(2.0, 3.0, -1.0),
            Value.listOf(3.0, -1.0, 0.5),
            Value.listOf(0.5, 1.0, 1.0),
            Value.listOf(1.0, 1.0, -1.0)
        )

        val yTargets: List<Value> = Value.listOf(1.0, -1.0, -1.0, 1.0) // desired targets

        // training loop
        Trainer().train(nn, xs, yTargets)

        val predictions = xs.map { x -> nn.call(x) }
        println("Predictions: $predictions")

        predictions.indices.forEach {
            Assert.assertEquals("Prediction $it should approximate target", yTargets[it].data, predictions[it][0].data, 0.3)
        }

    }
}