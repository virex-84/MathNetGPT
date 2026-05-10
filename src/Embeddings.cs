//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Llm.Core;

public class Embeddings : ILayer
{
    public Matrix<float> TokenEmbeddings { get; set; }
    public Matrix<float> PositionalEmbeddings { get; set; }
    private Matrix<float>? _cachedInput;
    private readonly Adam _tokenOptimizer;
    private readonly Adam _positionalOptimizer;

    public string LayerType => "Embeddings";

    public int Parameters =>
        (TokenEmbeddings.RowCount * TokenEmbeddings.ColumnCount) +
        (PositionalEmbeddings.RowCount * PositionalEmbeddings.ColumnCount);

    public Embeddings(Vocab vocab, int maxSeqLen)
    {
        TokenEmbeddings = InitEmbeddings(vocab.Words.Count, Constants.EMBEDDING_DIM);
        PositionalEmbeddings = InitPositionalEmbeddings(maxSeqLen, Constants.EMBEDDING_DIM);
        _tokenOptimizer = new Adam(vocab.Words.Count, Constants.EMBEDDING_DIM);
        _positionalOptimizer = new Adam(maxSeqLen, Constants.EMBEDDING_DIM);
    }

    private static Matrix<float> InitEmbeddings(int vocabSize, int embeddingDim)
    {
        var normal = new Normal(0.0, 0.02);
        return Matrix<float>.Build.Dense(vocabSize, embeddingDim, (i, j) => (float)normal.Sample());
    }

    private static Matrix<float> InitPositionalEmbeddings(int maxSeqLen, int embeddingDim)
    {
        var normal = new Normal(0.0, 0.02);
        return Matrix<float>.Build.Dense(maxSeqLen, embeddingDim, (i, j) => (float)normal.Sample());
    }

    public Matrix<float> Forward(Matrix<float> input)
    {
        _cachedInput = input;

        // OPT: избегаем Enumerate().Select().ToArray() Ч пр€мой доступ
        int seqLen = input.ColumnCount; // input shape [1, seqLen]
        var tokenIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokenIds[i] = (int)input[0, i];

        return EmbedTokens(tokenIds);
    }

    public Matrix<float> Backward(Matrix<float> grads, float learningRate)
    {
        if (_cachedInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int seqLen = _cachedInput.ColumnCount;
        var tokenIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokenIds[i] = (int)_cachedInput[0, i];

        int tokenRows = TokenEmbeddings.RowCount;
        int tokenCols = TokenEmbeddings.ColumnCount;
        int posRows = PositionalEmbeddings.RowCount;
        int posCols = PositionalEmbeddings.ColumnCount;

        var tokenGrads = Matrix<float>.Build.Dense(tokenRows, tokenCols);
        var positionalGrads = Matrix<float>.Build.Dense(posRows, posCols);

        for (int i = 0; i < tokenIds.Length; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId >= tokenRows)
                throw new IndexOutOfRangeException(
                    $"Token ID {tokenId} out of bounds for vocab size {tokenRows}");

            for (int j = 0; j < tokenCols; j++)
            {
                float g = grads[i, j];
                tokenGrads[tokenId, j] += g;
                positionalGrads[i, j] += g;
            }
        }

        _tokenOptimizer.Step(TokenEmbeddings, tokenGrads, learningRate);
        _positionalOptimizer.Step(PositionalEmbeddings, positionalGrads, learningRate);

        return grads;
    }

    private Matrix<float> EmbedTokens(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        int dim = TokenEmbeddings.ColumnCount;
        var result = Matrix<float>.Build.Dense(seqLen, dim);

        // OPT: пр€мое копирование по элементам вместо Row/SetRow
        for (int i = 0; i < seqLen; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId >= TokenEmbeddings.RowCount)
                throw new IndexOutOfRangeException(
                    $"Token ID {tokenId} out of bounds for vocab size {TokenEmbeddings.RowCount}");

            for (int j = 0; j < dim; j++)
                // token + positional embedding сразу в один проход
                result[i, j] = TokenEmbeddings[tokenId, j] + PositionalEmbeddings[i, j];
        }

        return result;
    }

    public List<SerializableMatrix> GetParameters()
    {
        var parameters = new List<SerializableMatrix>
        {
            SerializableMatrix.FromMatrix(TokenEmbeddings),
            SerializableMatrix.FromMatrix(PositionalEmbeddings),
        };
        parameters.AddRange(_tokenOptimizer.GetParameters());
        parameters.AddRange(_positionalOptimizer.GetParameters());
        return parameters;
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        TokenEmbeddings = parameters.Dequeue().ToMatrix();
        PositionalEmbeddings = parameters.Dequeue().ToMatrix();
        _tokenOptimizer.SetParameters(parameters);
        _positionalOptimizer.SetParameters(parameters);
    }
}