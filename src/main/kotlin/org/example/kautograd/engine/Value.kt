package org.example.kautograd.engine

import kotlin.math.exp
import kotlin.random.Random

class Value(var data: Double, private val previous: Collection<Value>? = emptyList()) {
    private var gradient = 0.0
    private var backwards = { }

    companion object {
        fun random(): Value {
            return Value(Random.nextDouble(-1.0, 1.0))
        }

        fun listOf(vararg d: Double): List<Value> {
            return d.map { d -> Value(d) }
        }

        fun tanh(value: Value): Value {
            val x: Double = value.data
            val t = (exp(2 * x) - 1) / (exp(2 * x) + 1)

            val out = Value(t, listOf(value))

            out.backwards = {
                value.gradient += (1 - t * t) * out.gradient
            }
            return out
        }
    }

    operator fun plus(other: Value): Value {
        val out = Value(data + other.data, listOf(this, other))
        out.backwards = {
            gradient += out.gradient
            other.gradient += out.gradient
        }
        return out
    }

    operator fun minus(other: Value): Value {
        return this + (Value(-1.0) * other);
    }

    operator fun times(other: Value): Value {
        val out = Value(data * other.data, listOf(this, other))
        out.backwards = {
            this.gradient += other.data * out.gradient
            other.gradient += this.data * out.gradient
        }
        return out
    }

    override fun toString(): String {
        return "Value($data, $gradient)"
    }

    fun resetGrad() {
        gradient = 0.0
    }

    fun backwardsPropagation() {
        var topo = mutableListOf<Value>()
        var visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (!visited.contains(v)) {
                visited.add(v)
                for (child in v.previous!!) {
                    buildTopo(child)
                }
                topo.add(v)
            }
        }
        buildTopo(this)

        gradient = 1.0
        for (v in topo.reversed()) {
            v.backwards.invoke()
        }

    }

    fun adjustWithGradient(factor: Double) {
        data += -factor * gradient
    }


}