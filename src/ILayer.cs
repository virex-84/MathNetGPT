//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;

namespace Llm.Core;

public interface ILayer
{
    string LayerType { get; }
    Matrix<float> Forward(Matrix<float> input);
    Matrix<float> Backward(Matrix<float> grads, float learningRate);
    int Parameters { get; }
    List<SerializableMatrix> GetParameters();
    void SetParameters(Queue<SerializableMatrix> parameters);
}
