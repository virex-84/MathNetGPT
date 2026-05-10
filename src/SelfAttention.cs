//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Llm.Core;

public class SelfAttention : ILayer
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _scale;      // OPT: предвычислен в конструкторе
    private readonly float _invScale;   // OPT: 1/scale для деления → умножение

    private Matrix<float> _wQ;
    private Matrix<float> _wK;
    private Matrix<float> _wV;
    private Matrix<float> _wO;

    private Matrix<float>? _cachedInput;
    private Matrix<float>? _cachedQ;
    private Matrix<float>? _cachedK;
    private Matrix<float>? _cachedV;
    private Matrix<float>[]? _cachedHeadWeights;
    private Matrix<float>? _cachedAllHeadsOutput;

    private readonly Adam _optimizerWQ;
    private readonly Adam _optimizerWK;
    private readonly Adam _optimizerWV;
    private readonly Adam _optimizerWO;

    public string LayerType => "SelfAttention";

    public int Parameters =>
        (_wQ.RowCount * _wQ.ColumnCount) +
        (_wK.RowCount * _wK.ColumnCount) +
        (_wV.RowCount * _wV.ColumnCount) +
        (_wO.RowCount * _wO.ColumnCount);

    public SelfAttention(int embeddingDim, int numHeads = 4)
    {
        if (embeddingDim % numHeads != 0)
            throw new ArgumentException(
                $"embeddingDim ({embeddingDim}) must be divisible by numHeads ({numHeads})");

        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        // OPT: предвычисляем scale один раз
        _scale = MathF.Sqrt(_headDim);
        _invScale = 1.0f / _scale;

        var std = Math.Sqrt(2.0 / embeddingDim);
        _wQ = Matrix<float>.Build.Random(embeddingDim, embeddingDim, new Normal(0, std));
        _wK = Matrix<float>.Build.Random(embeddingDim, embeddingDim, new Normal(0, std));
        _wV = Matrix<float>.Build.Random(embeddingDim, embeddingDim, new Normal(0, std));
        _wO = Matrix<float>.Build.Random(embeddingDim, embeddingDim, new Normal(0, std));

        _optimizerWQ = new Adam(embeddingDim, embeddingDim);
        _optimizerWK = new Adam(embeddingDim, embeddingDim);
        _optimizerWV = new Adam(embeddingDim, embeddingDim);
        _optimizerWO = new Adam(embeddingDim, embeddingDim);
    }

    public Matrix<float> Forward(Matrix<float> input)
    {
        _cachedInput = input;
        int seqLen = input.RowCount;

        var q = input.Multiply(_wQ);
        var k = input.Multiply(_wK);
        var v = input.Multiply(_wV);

        _cachedQ = q;
        _cachedK = k;
        _cachedV = v;

        var allHeadsOutput = Matrix<float>.Build.Dense(seqLen, _embeddingDim);
        _cachedHeadWeights = new Matrix<float>[_numHeads];

        for (int h = 0; h < _numHeads; h++)
        {
            int startCol = h * _headDim;

            var qH = q.SubMatrix(0, seqLen, startCol, _headDim);
            var kH = k.SubMatrix(0, seqLen, startCol, _headDim);
            var vH = v.SubMatrix(0, seqLen, startCol, _headDim);

            // OPT: умножение на _invScale вместо деления на _scale
            var scores = qH.Multiply(kH.Transpose());
            ScaleInPlace(scores, _invScale);

            // Causal mask — только верхний треугольник
            ApplyCausalMask(scores);

            var weights = SoftmaxRows(scores);
            _cachedHeadWeights[h] = weights;

            var headOut = weights.Multiply(vH);
            allHeadsOutput.SetSubMatrix(0, startCol, headOut);
        }

        _cachedAllHeadsOutput = allHeadsOutput;
        return allHeadsOutput.Multiply(_wO);
    }

    public Matrix<float> Backward(Matrix<float> grads, float lr)
    {
        var input = _cachedInput ?? throw new InvalidOperationException(
            "Forward pass must be called before backward pass.");

        var q = _cachedQ!;
        var k = _cachedK!;
        var v = _cachedV!;
        var allHeadsOutput = _cachedAllHeadsOutput!;
        var headWeights = _cachedHeadWeights!;

        int seqLen = input.RowCount;

        var gradWO = allHeadsOutput.TransposeThisAndMultiply(grads);
        var gradAllHeads = grads.Multiply(_wO.Transpose());

        var gradQ = Matrix<float>.Build.Dense(seqLen, _embeddingDim);
        var gradK = Matrix<float>.Build.Dense(seqLen, _embeddingDim);
        var gradV = Matrix<float>.Build.Dense(seqLen, _embeddingDim);

        for (int h = 0; h < _numHeads; h++)
        {
            int startCol = h * _headDim;
            var qH = q.SubMatrix(0, seqLen, startCol, _headDim);
            var kH = k.SubMatrix(0, seqLen, startCol, _headDim);
            var vH = v.SubMatrix(0, seqLen, startCol, _headDim);

            var gradHeadOut = gradAllHeads.SubMatrix(0, seqLen, startCol, _headDim);
            var attnW = headWeights[h];

            var gradVH = attnW.TransposeThisAndMultiply(gradHeadOut);
            var gradAttnW = gradHeadOut.Multiply(vH.Transpose());
            var gradScores = SoftmaxBackward(attnW, gradAttnW);

            // OPT: умножение на _invScale вместо деления
            var gradQH = gradScores.Multiply(kH);
            ScaleInPlace(gradQH, _invScale);

            var gradKH = gradScores.TransposeThisAndMultiply(qH);
            ScaleInPlace(gradKH, _invScale);

            gradQ.SetSubMatrix(0, startCol, gradQH);
            gradK.SetSubMatrix(0, startCol, gradKH);
            gradV.SetSubMatrix(0, startCol, gradVH);
        }

        var gradWQ = input.TransposeThisAndMultiply(gradQ);
        var gradWK = input.TransposeThisAndMultiply(gradK);
        var gradWV = input.TransposeThisAndMultiply(gradV);

        var gradInput =
            gradQ.Multiply(_wQ.Transpose()) +
            gradK.Multiply(_wK.Transpose()) +
            gradV.Multiply(_wV.Transpose());

        _optimizerWQ.Step(_wQ, gradWQ, lr);
        _optimizerWK.Step(_wK, gradWK, lr);
        _optimizerWV.Step(_wV, gradWV, lr);
        _optimizerWO.Step(_wO, gradWO, lr);

        return gradInput;
    }

    /// <summary>
    /// OPT: scale in-place — не создаём новую матрицу
    /// </summary>
    private static void ScaleInPlace(Matrix<float> m, float scale)
    {
        int rows = m.RowCount;
        int cols = m.ColumnCount;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] *= scale;
    }

    /// <summary>
    /// OPT: causal mask — только верхний треугольник, без проверки диагонали
    /// </summary>
    private static void ApplyCausalMask(Matrix<float> scores)
    {
        int n = scores.RowCount;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                scores[i, j] = float.NegativeInfinity;
    }

    /// <summary>
    /// OPT: Softmax построчно без создания промежуточных Vector через Row()
    /// </summary>
    private static Matrix<float> SoftmaxRows(Matrix<float> matrix)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
        var result = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            // Находим max в строке
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                float val = matrix[i, j];
                if (val > maxVal) maxVal = val;
            }

            // Вычисляем exp и sum
            float sum = 0f;
            for (int j = 0; j < cols; j++)
            {
                float e = MathF.Exp(matrix[i, j] - maxVal);
                result[i, j] = e;
                sum += e;
            }

            // Нормализуем
            float invSum = 1.0f / sum;
            for (int j = 0; j < cols; j++)
                result[i, j] *= invSum;
        }

        return result;
    }

    /// <summary>
    /// OPT: SoftmaxBackward без Row() аллокаций
    /// </summary>
    private static Matrix<float> SoftmaxBackward(
        Matrix<float> softmaxOutput,
        Matrix<float> gradOutput)
    {
        int rows = softmaxOutput.RowCount;
        int cols = softmaxOutput.ColumnCount;
        var gradInput = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            // Вычисляем dot product inline без Row()
            float dot = 0f;
            for (int j = 0; j < cols; j++)
                dot += softmaxOutput[i, j] * gradOutput[i, j];

            for (int j = 0; j < cols; j++)
                gradInput[i, j] = softmaxOutput[i, j] * (gradOutput[i, j] - dot);
        }

        return gradInput;
    }

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>
            {
                SerializableMatrix.FromMatrix(_wQ),
                SerializableMatrix.FromMatrix(_wK),
                SerializableMatrix.FromMatrix(_wV),
                SerializableMatrix.FromMatrix(_wO),
            };
        parameters.AddRange(_optimizerWQ.GetParameters());
        parameters.AddRange(_optimizerWK.GetParameters());
        parameters.AddRange(_optimizerWV.GetParameters());
        parameters.AddRange(_optimizerWO.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        _wQ = parameters.Dequeue().ToMatrix();
        _wK = parameters.Dequeue().ToMatrix();
        _wV = parameters.Dequeue().ToMatrix();
        _wO = parameters.Dequeue().ToMatrix();
        _optimizerWQ.SetParameters(parameters);
        _optimizerWK.SetParameters(parameters);
        _optimizerWV.SetParameters(parameters);
        _optimizerWO.SetParameters(parameters);
    }
}
