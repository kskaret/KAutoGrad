import org.example.kautograd.engine.Value
import org.example.kautograd.nn.Layer
import org.example.kautograd.nn.MLP
import org.example.kautograd.nn.Neuron
import org.example.kautograd.nn.Trainer

fun main() {
    println("Hello Neural Net!")

    val v1 = Value(1.0)
    val v2 = Value(2.0)
    val v3 = v1 + v2

    println(v3)
}



