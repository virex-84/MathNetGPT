//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;

namespace Llm.Core;

public class TransformerBlock : ILayer
{
    private readonly SelfAttention _attention;
    private readonly FeedForward _feedForward;
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;

    public TransformerBlock(int embeddingDim, int hiddenDim, int numHeads = 4)
    {
        _attention = new SelfAttention(embeddingDim, numHeads);
        _feedForward = new FeedForward(embeddingDim, hiddenDim);
        _norm1 = new LayerNorm(embeddingDim);
        _norm2 = new LayerNorm(embeddingDim);
    }

    public string LayerType => "TransformerBlock";

    // Pre-Norm: x1 = x + Attention(Norm1(x)), x2 = x1 + FFN(Norm2(x1))
    public Matrix<float> Forward(Matrix<float> input)
    {
        var norm1Out = _norm1.Forward(input);
        var attnOut = _attention.Forward(norm1Out);
        var x1 = attnOut + input;

        var norm2Out = _norm2.Forward(x1);
        var ffnOut = _feedForward.Forward(norm2Out);
        return ffnOut + x1;
    }

    public Matrix<float> Backward(Matrix<float> grads, float lr)
    {
        // Backward: x2 = ffnOut + x1
        var gradFfn = _feedForward.Backward(grads, lr);
        var gradNorm2 = _norm2.Backward(gradFfn, lr);
        var gradX1 = gradNorm2 + grads;

        // Backward: x1 = attnOut + input
        var gradAttn = _attention.Backward(gradX1, lr);
        var gradNorm1 = _norm1.Backward(gradAttn, lr);
        var gradInput = gradNorm1 + gradX1;

        return gradInput;
    }

    public int Parameters =>
        _attention.Parameters + _feedForward.Parameters +
        _norm1.Parameters + _norm2.Parameters;

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>();
        parameters.AddRange(_attention.GetParameters());
        parameters.AddRange(_norm1.GetParameters());
        parameters.AddRange(_feedForward.GetParameters());
        parameters.AddRange(_norm2.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        _attention.SetParameters(parameters);
        _norm1.SetParameters(parameters);
        _feedForward.SetParameters(parameters);
        _norm2.SetParameters(parameters);
    }
}