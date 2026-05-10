//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;

namespace Llm.Core;

public class LayerNorm : ILayer
{
    private readonly float _epsilon = 1e-5f;
    private Matrix<float> _gamma;
    private Matrix<float> _beta;

    private Matrix<float>? _cachedInput;
    private Matrix<float>? _cachedNormalized;
    private Vector<float>? _cachedInvStd;

    private float[]? _meanBuf;
    private float[]? _invStdBuf;

    // OPT: кэшируем gamma/beta как float[] — обновляем только после Step
    private float[]? _gammaCached;
    private float[]? _betaCached;
    // Буфер для dxHat чтобы не вычислять дважды в Backward
    private float[]? _dxHatBuf;

    private readonly Adam _optimizerGamma;
    private readonly Adam _optimizerBeta;

    public string LayerType => "LayerNorm";
    public int Parameters => (_gamma.RowCount * _gamma.ColumnCount) +
                             (_beta.RowCount * _beta.ColumnCount);

    public LayerNorm(int embeddingDim)
    {
        _gamma = Matrix<float>.Build.Dense(1, embeddingDim, 1.0f);
        _beta = Matrix<float>.Build.Dense(1, embeddingDim, 0.0f);
        _optimizerGamma = new Adam(1, embeddingDim);
        _optimizerBeta = new Adam(1, embeddingDim);
        // Инициализируем кэш сразу
        _gammaCached = Enumerable.Repeat(1.0f, embeddingDim).ToArray();
        _betaCached = new float[embeddingDim];
    }

    public Matrix<float> Forward(Matrix<float> input)
    {
        int rows = input.RowCount;
        int cols = input.ColumnCount;

        _cachedInput = input;
        EnsureBuffers(rows, cols);

        var mean = _meanBuf!;
        var invStd = _invStdBuf!;
        var gammaArr = _gammaCached!;
        var betaArr = _betaCached!;

        if (_cachedNormalized == null ||
            _cachedNormalized.RowCount != rows ||
            _cachedNormalized.ColumnCount != cols)
        {
            _cachedNormalized = Matrix<float>.Build.Dense(rows, cols);
        }

        // Вычисляем mean
        for (int i = 0; i < rows; i++)
        {
            float sum = 0f;
            for (int j = 0; j < cols; j++)
                sum += input[i, j];
            mean[i] = sum / cols;
        }

        // Вычисляем variance и invStd
        for (int i = 0; i < rows; i++)
        {
            float m = mean[i];
            float variance = 0f;
            for (int j = 0; j < cols; j++)
            {
                float diff = input[i, j] - m;
                variance += diff * diff;
            }
            invStd[i] = 1.0f / MathF.Sqrt(variance / cols + _epsilon);
        }

        // Сохраняем invStd
        if (_cachedInvStd == null || _cachedInvStd.Count != rows)
            _cachedInvStd = Vector<float>.Build.Dense(rows);
        for (int i = 0; i < rows; i++)
            _cachedInvStd[i] = invStd[i];

        var output = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            float m = mean[i];
            float inv = invStd[i];
            for (int j = 0; j < cols; j++)
            {
                float norm = (input[i, j] - m) * inv;
                _cachedNormalized[i, j] = norm;
                output[i, j] = norm * gammaArr[j] + betaArr[j];
            }
        }

        return output;
    }

    public Matrix<float> Backward(Matrix<float> grads, float lr)
    {
        if (_cachedInput == null || _cachedNormalized == null || _cachedInvStd == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var xHat = _cachedNormalized;
        var invStd = _cachedInvStd;
        var gammaArr = _gammaCached!;

        int rows = _cachedInput.RowCount;
        int cols = _cachedInput.ColumnCount;
        float n = cols;

        var gradGammaArr = new float[cols];
        var gradBetaArr = new float[cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                gradGammaArr[j] += grads[i, j] * xHat[i, j];
                gradBetaArr[j] += grads[i, j];
            }

        var gradGamma = Matrix<float>.Build.Dense(1, cols, gradGammaArr);
        var gradBeta = Matrix<float>.Build.Dense(1, cols, gradBetaArr);

        var gradInput = Matrix<float>.Build.Dense(rows, cols);

        // OPT: буфер для dxHat — не вычисляем дважды
        if (_dxHatBuf == null || _dxHatBuf.Length < cols)
            _dxHatBuf = new float[cols];

        var dxHatBuf = _dxHatBuf;

        for (int i = 0; i < rows; i++)
        {
            float inv = invStd[i];
            float sum1 = 0f, sum2 = 0f;

            // Первый проход — вычисляем suммы и сохраняем dxHat в буфер
            for (int j = 0; j < cols; j++)
            {
                float dxHat = grads[i, j] * gammaArr[j];
                dxHatBuf[j] = dxHat;
                sum1 += dxHat;
                sum2 += dxHat * xHat[i, j];
            }

            // Второй проход — используем сохранённые dxHat из буфера
            float scale = inv / n;
            for (int j = 0; j < cols; j++)
            {
                gradInput[i, j] = scale * (n * dxHatBuf[j] - sum1 - xHat[i, j] * sum2);
            }
        }

        _optimizerGamma.Step(_gamma, gradGamma, lr);
        _optimizerBeta.Step(_beta, gradBeta, lr);

        // OPT: обновляем кэш gamma/beta после шага оптимизатора
        RefreshGammaBetaCache();

        return gradInput;
    }

    private void RefreshGammaBetaCache()
    {
        int cols = _gamma.ColumnCount;
        if (_gammaCached == null || _gammaCached.Length != cols)
        {
            _gammaCached = new float[cols];
            _betaCached = new float[cols];
        }
        for (int j = 0; j < cols; j++)
        {
            _gammaCached[j] = _gamma[0, j];
            _betaCached![j] = _beta[0, j];
        }
    }

    private void EnsureBuffers(int rows, int cols)
    {
        if (_meanBuf == null || _meanBuf.Length < rows)
        {
            _meanBuf = new float[rows];
            _invStdBuf = new float[rows];
        }

        // Инициализируем gamma/beta кэш если нужно
        if (_gammaCached == null || _gammaCached.Length != cols)
            RefreshGammaBetaCache();
    }

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>
        {
            SerializableMatrix.FromMatrix(_gamma),
            SerializableMatrix.FromMatrix(_beta),
        };
        parameters.AddRange(_optimizerGamma.GetParameters());
        parameters.AddRange(_optimizerBeta.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        _gamma = parameters.Dequeue().ToMatrix();
        _beta = parameters.Dequeue().ToMatrix();
        _optimizerGamma.SetParameters(parameters);
        _optimizerBeta.SetParameters(parameters);
        // OPT: обновляем кэш после загрузки параметров
        RefreshGammaBetaCache();
    }
}