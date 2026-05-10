//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Llm.Core;

public class FeedForward : ILayer
{
    private Matrix<float> _w1;
    private Matrix<float> _b1;
    private Matrix<float> _w2;
    private Matrix<float> _b2;

    private Matrix<float>? _input;
    private Matrix<float>? _hiddenPreActivation;
    private Matrix<float>? _hiddenPostActivation;

    private readonly Adam _optimizerW1;
    private readonly Adam _optimizerB1;
    private readonly Adam _optimizerW2;
    private readonly Adam _optimizerB2;

    private const float SqrtTwoOverPi = 0.7978845608f;
    private const float GeluCoeff = 0.044715f;

    // OPT: кэшируем bias как float[] для быстрого доступа
    private float[]? _b1Cache;
    private float[]? _b2Cache;

    public string LayerType => "FeedForward";

    public int Parameters =>
        (_b1.RowCount * _b1.ColumnCount) +
        (_b2.RowCount * _b2.ColumnCount) +
        (_w1.RowCount * _w1.ColumnCount) +
        (_w2.RowCount * _w2.ColumnCount);

    public FeedForward(int embeddingDim, int hiddenDim)
    {
        var stdW1 = Math.Sqrt(2.0 / embeddingDim);
        var stdW2 = Math.Sqrt(2.0 / hiddenDim);

        _w1 = Matrix<float>.Build.Random(embeddingDim, hiddenDim, new Normal(0, stdW1));
        _b1 = Matrix<float>.Build.Dense(1, hiddenDim, 0);
        _w2 = Matrix<float>.Build.Random(hiddenDim, embeddingDim, new Normal(0, stdW2));
        _b2 = Matrix<float>.Build.Dense(1, embeddingDim, 0);

        _optimizerW1 = new Adam(embeddingDim, hiddenDim);
        _optimizerB1 = new Adam(1, hiddenDim);
        _optimizerW2 = new Adam(hiddenDim, embeddingDim);
        _optimizerB2 = new Adam(1, embeddingDim);

        // Инициализируем кэш (bias = 0, так что можно не заполнять)
        _b1Cache = new float[hiddenDim];
        _b2Cache = new float[embeddingDim];
    }

    public Matrix<float> Forward(Matrix<float> input)
    {
        _input = input;

        _hiddenPreActivation = input.Multiply(_w1);
        AddBiasInPlace(_hiddenPreActivation, _b1Cache!);

        // OPT: заменяем Map(Gelu) на ручной цикл — нет overhead делегата
        _hiddenPostActivation = ApplyGeluInPlace(_hiddenPreActivation);

        var output = _hiddenPostActivation.Multiply(_w2);
        AddBiasInPlace(output, _b2Cache!);

        return output;
    }

    public Matrix<float> Backward(Matrix<float> grads, float lr)
    {
        if (_input == null || _hiddenPreActivation == null || _hiddenPostActivation == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var gradW2 = _hiddenPostActivation.TransposeThisAndMultiply(grads);
        var gradB2 = grads.ColumnSums().ToRowMatrix();

        var gradHiddenPostActivation = grads.Multiply(_w2.Transpose());

        // OPT: объединяем GeluDerivative и PointwiseMultiply в один цикл
        var gradHiddenPreActivation = ApplyGeluDerivativeAndMultiply(
            _hiddenPreActivation, gradHiddenPostActivation);

        var gradW1 = _input.TransposeThisAndMultiply(gradHiddenPreActivation);
        var gradB1 = gradHiddenPreActivation.ColumnSums().ToRowMatrix();

        var gradInput = gradHiddenPreActivation.Multiply(_w1.Transpose());

        _optimizerW2.Step(_w2, gradW2, lr);
        _optimizerB2.Step(_b2, gradB2, lr);
        _optimizerW1.Step(_w1, gradW1, lr);
        _optimizerB1.Step(_b1, gradB1, lr);

        // OPT: обновляем bias кэш
        RefreshBiasCache();

        return gradInput;
    }

    /// <summary>
    /// OPT: bias как float[] — прямой доступ без Matrix индексатора
    /// </summary>
    private static void AddBiasInPlace(Matrix<float> matrix, float[] bias)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] += bias[j];
    }

    /// <summary>
    /// OPT: GELU без Map() — нет аллокации делегата и boxing
    /// Создаёт новую матрицу с результатом (нужна для кэша в Backward)
    /// </summary>
    private static Matrix<float> ApplyGeluInPlace(Matrix<float> source)
    {
        int rows = source.RowCount;
        int cols = source.ColumnCount;
        var result = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                float x = source[i, j];
                float inner = SqrtTwoOverPi * (x + GeluCoeff * x * x * x);
                result[i, j] = 0.5f * x * (1.0f + MathF.Tanh(inner));
            }

        return result;
    }

    /// <summary>
    /// OPT: вычисляем GeluDerivative и PointwiseMultiply за один проход
    /// вместо двух отдельных операций Map + PointwiseMultiply
    /// </summary>
    private static Matrix<float> ApplyGeluDerivativeAndMultiply(
        Matrix<float> preActivation, Matrix<float> upstream)
    {
        int rows = preActivation.RowCount;
        int cols = preActivation.ColumnCount;
        var result = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                float x = preActivation[i, j];
                float inner = SqrtTwoOverPi * (x + GeluCoeff * x * x * x);
                float tanh = MathF.Tanh(inner);
                float cdf = 0.5f * (1.0f + tanh);
                float innerDeriv = SqrtTwoOverPi * (1.0f + 3.0f * GeluCoeff * x * x);
                float pdf = 0.5f * x * (1.0f - tanh * tanh) * innerDeriv;
                result[i, j] = (cdf + pdf) * upstream[i, j];
            }

        return result;
    }

    private void RefreshBiasCache()
    {
        if (_b1Cache == null || _b1Cache.Length != _b1.ColumnCount)
            _b1Cache = new float[_b1.ColumnCount];
        if (_b2Cache == null || _b2Cache.Length != _b2.ColumnCount)
            _b2Cache = new float[_b2.ColumnCount];

        for (int j = 0; j < _b1.ColumnCount; j++)
            _b1Cache[j] = _b1[0, j];
        for (int j = 0; j < _b2.ColumnCount; j++)
            _b2Cache[j] = _b2[0, j];
    }

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>
        {
            SerializableMatrix.FromMatrix(_w1),
            SerializableMatrix.FromMatrix(_b1),
            SerializableMatrix.FromMatrix(_w2),
            SerializableMatrix.FromMatrix(_b2),
        };
        parameters.AddRange(_optimizerW1.GetParameters());
        parameters.AddRange(_optimizerB1.GetParameters());
        parameters.AddRange(_optimizerW2.GetParameters());
        parameters.AddRange(_optimizerB2.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        _w1 = parameters.Dequeue().ToMatrix();
        _b1 = parameters.Dequeue().ToMatrix();
        _w2 = parameters.Dequeue().ToMatrix();
        _b2 = parameters.Dequeue().ToMatrix();
        _optimizerW1.SetParameters(parameters);
        _optimizerB1.SetParameters(parameters);
        _optimizerW2.SetParameters(parameters);
        _optimizerB2.SetParameters(parameters);
        RefreshBiasCache();
    }
}