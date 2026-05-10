//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Llm.Core;

public class OutputProjection : ILayer
{
    private Matrix<float> _wOut;
    private Matrix<float> _bOut;
    private readonly Adam _optimizerW;
    private readonly Adam _optimizerB;
    private Matrix<float>? _cachedInput;

    // OPT: кэш bias как float[]
    private float[]? _bOutCache;

    public string LayerType => "OutputProjection";

    public int Parameters =>
        (_wOut.RowCount * _wOut.ColumnCount) +
        (_bOut.RowCount * _bOut.ColumnCount);

    public OutputProjection(int embeddingDim, int vocabSize)
    {
        var std = Math.Sqrt(2.0 / embeddingDim);
        _wOut = Matrix<float>.Build.Random(embeddingDim, vocabSize, new Normal(0, std));
        _bOut = Matrix<float>.Build.Dense(1, vocabSize, 0);
        _optimizerW = new Adam(embeddingDim, vocabSize);
        _optimizerB = new Adam(1, vocabSize);
        _bOutCache = new float[vocabSize]; // всё нули, bias = 0 изначально
    }

    public Matrix<float> Forward(Matrix<float> input)
    {
        _cachedInput = input;
        var output = input.Multiply(_wOut);

        // OPT: используем float[] кэш вместо _bOut[0,j]
        var bCache = _bOutCache!;
        int rows = output.RowCount;
        int cols = output.ColumnCount;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                output[i, j] += bCache[j];

        return output;
    }

    public Matrix<float> Backward(Matrix<float> grads, float lr)
    {
        if (_cachedInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var gradWOut = _cachedInput.TransposeThisAndMultiply(grads);

        // OPT: вычисляем gradBOut inline без ColumnSums().ToRowMatrix()
        int rows = grads.RowCount;
        int cols = grads.ColumnCount;
        float invRows = 1.0f / rows;
        var gradBOutArr = new float[cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                gradBOutArr[j] += grads[i, j];

        // Усредняем если нужно
        if (rows > 1)
            for (int j = 0; j < cols; j++)
                gradBOutArr[j] *= invRows;

        var gradBOut = Matrix<float>.Build.Dense(1, cols, gradBOutArr);
        var gradInput = grads.Multiply(_wOut.Transpose());

        _optimizerW.Step(_wOut, gradWOut, lr);
        _optimizerB.Step(_bOut, gradBOut, lr);

        // OPT: обновляем кэш bias
        RefreshBiasCache();

        return gradInput;
    }

    private void RefreshBiasCache()
    {
        int cols = _bOut.ColumnCount;
        if (_bOutCache == null || _bOutCache.Length != cols)
            _bOutCache = new float[cols];
        for (int j = 0; j < cols; j++)
            _bOutCache[j] = _bOut[0, j];
    }

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>
        {
            SerializableMatrix.FromMatrix(_wOut),
            SerializableMatrix.FromMatrix(_bOut),
        };
        parameters.AddRange(_optimizerW.GetParameters());
        parameters.AddRange(_optimizerB.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        _wOut = parameters.Dequeue().ToMatrix();
        _bOut = parameters.Dequeue().ToMatrix();
        _optimizerW.SetParameters(parameters);
        _optimizerB.SetParameters(parameters);
        RefreshBiasCache();
    }
}