//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MessagePack;

namespace Llm.Core;

public class PredictionResult
{
    public string Text { get; set; } = string.Empty;
    public float AverageConfidence { get; set; }
    public float MinConfidence { get; set; }
    public float Entropy { get; set; }
    public bool WasRejected { get; set; }
    public List<float> StepConfidences { get; set; } = new();
}

public class LLM
{
    public Vocab vocab { get; private set; }
    public List<ILayer> network { get; private set; }
    public int MaxSeqLen { get; private set; }

    private int _eosTokenId;
    private List<ILayer> _reversedNetwork = new();

    public LLM(int maxSeqLen)
    {
        MaxSeqLen = maxSeqLen;
        vocab = new Vocab();
        _eosTokenId = vocab.EosTokenId;
        var vocabSize = vocab.Words.Count;

        network = new List<ILayer>
        {
            new Embeddings(vocab, maxSeqLen),
            new TransformerBlock(Constants.EMBEDDING_DIM, Constants.HIDDEN_DIM),
            new OutputProjection(Constants.EMBEDDING_DIM, vocabSize)
        };
        RebuildReversedNetwork();
    }

    public LLM(Vocab vocab, int maxSeqLen, List<ILayer> network)
    {
        this.vocab = vocab;
        MaxSeqLen = maxSeqLen;
        this.network = network;
        _eosTokenId = vocab.EosTokenId;
        RebuildReversedNetwork();
    }

    private void RebuildReversedNetwork()
    {
        _reversedNetwork = new List<ILayer>(network);
        _reversedNetwork.Reverse();
    }

    public void Save(string path)
    {
        var allParameters = new List<SerializableMatrix>();
        foreach (var layer in network)
            allParameters.AddRange(layer.GetParameters());

        int transformerBlockCount = network.Count(l => l is TransformerBlock);

        var modelState = new ModelState
        {
            AllParameters = allParameters,
            VocabWords = this.vocab.Words,
            TransformerBlockCount = transformerBlockCount,
            EmbeddingDim = Constants.EMBEDDING_DIM,
            HiddenDim = Constants.HIDDEN_DIM,
            MaxSeqLen = GetMaxSeqLenFromNetwork()
        };

        var bytes = MessagePackSerializer.Serialize(modelState);
        File.WriteAllBytes(path, bytes);
    }

    public void Load(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var modelState = MessagePackSerializer.Deserialize<ModelState>(bytes);

        this.vocab = new Vocab(modelState.VocabWords);
        // FIX: vocab создаётся через конструктор — CacheSpecialTokenIds уже вызван
        _eosTokenId = this.vocab.EosTokenId;
        this.MaxSeqLen = modelState.MaxSeqLen;
        this.network = BuildNetwork(modelState);
        RebuildReversedNetwork();
        LoadParameters(modelState.AllParameters);
    }

    public static LLM LoadModel(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var modelState = MessagePackSerializer.Deserialize<ModelState>(bytes);

        // FIX: Vocab создаётся через конструктор (IEnumerable<string>),
        // который сам вызывает CacheSpecialTokenIds — PostDeserialize не нужен
        var vocab = new Vocab(modelState.VocabWords);
        var network = BuildNetwork(modelState);
        var llm = new LLM(vocab, modelState.MaxSeqLen, network);
        llm.LoadParameters(modelState.AllParameters);
        return llm;
    }

    private static List<ILayer> BuildNetwork(ModelState modelState)
    {
        var network = new List<ILayer>
        {
            new Embeddings(new Vocab(modelState.VocabWords), modelState.MaxSeqLen)
        };

        for (int i = 0; i < modelState.TransformerBlockCount; i++)
            network.Add(new TransformerBlock(
                modelState.EmbeddingDim, modelState.HiddenDim));

        network.Add(new OutputProjection(
            modelState.EmbeddingDim, modelState.VocabWords.Count));

        return network;
    }

    private int GetMaxSeqLenFromNetwork()
    {
        foreach (var layer in network)
            if (layer is Embeddings embeddings)
                return embeddings.PositionalEmbeddings.RowCount;
        return Constants.MAX_SEQ_LEN;
    }

    private void LoadParameters(List<SerializableMatrix> allParameters)
    {
        var parameters = new Queue<SerializableMatrix>(allParameters);
        int expectedCount = allParameters.Count;

        foreach (var layer in network)
            layer.SetParameters(parameters);

        if (parameters.Count != 0)
            throw new InvalidOperationException(
                $"Unused parameters after loading: {parameters.Count}. " +
                $"Expected: {expectedCount}. Model architecture mismatch.");
    }

    public string NetworkDescription() =>
        string.Join(", ", network.Select(layer => layer.LayerType));

    public int TotalParameters() =>
        network.Sum(layer => layer.Parameters);

    public PredictionResult PredictWithConfidence(
        string text, float confidenceThreshold = 0.7f)
    {
        var outputTokens = ForwardWithConfidence(text, out var forwardResult);

        // FIX: убрано бессмысленное self-присваивание
        // forwardResult.StepConfidences = forwardResult.StepConfidences;

        if (outputTokens.Count == 0)
        {
            forwardResult.Text = string.Empty;
            forwardResult.WasRejected = true;
            return forwardResult;
        }

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < outputTokens.Count; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(vocab.DecodeToken(outputTokens[i]));
        }

        forwardResult.Text = sb.ToString();
        forwardResult.WasRejected = false;
        return forwardResult;
    }

    private List<int> ForwardWithConfidence(
        string text, out PredictionResult result)
    {
        result = new PredictionResult();
        var tokenized = Tokenizer.Tokenize(vocab, text);
        var outputTokens = new List<int>();
        var stepConfidences = new List<float>();

        if (tokenized.Count == 0 || tokenized.Count >= MaxSeqLen)
            return outputTokens;

        int maxNewTokens = MaxSeqLen - tokenized.Count;

        // FIX: переиспользуем массив токенов, расширяя по мере необходимости
        // вместо создания нового на каждом шаге
        var tokenArr = new float[MaxSeqLen];

        for (int i = 0; i < maxNewTokens; i++)
        {
            if (outputTokens.Count >= MaxSeqLen - 1)
                break;

            int currentLen = tokenized.Count;

            // FIX: заполняем существующий массив вместо создания нового
            for (int k = 0; k < currentLen; k++)
                tokenArr[k] = tokenized[k];

            // FIX: создаём матрицу нужного размера (view на часть массива)
            var tokenInput = Matrix<float>.Build.Dense(1, currentLen,
                (r, c) => tokenArr[c]);

            var input = tokenInput;
            foreach (var layer in network)
                input = layer.Forward(input);

            if (input.RowCount == 0) break;

            var lastLogit = input.Row(input.RowCount - 1).ToRowMatrix();
            var probs = Softmax(lastLogit);
            var (nextToken, confidence) = DecodeWithConfidence(probs.Row(0));

            stepConfidences.Add(confidence);
            outputTokens.Add(nextToken);
            tokenized.Add(nextToken);

            if (nextToken == _eosTokenId)
                break;
        }

        if (stepConfidences.Count > 0)
        {
            result.AverageConfidence = stepConfidences.Average();
            result.MinConfidence = stepConfidences.Min();
            result.StepConfidences = stepConfidences;
        }

        return outputTokens;
    }

    public void Train(List<string> data, int epochs, float lr)
    {
        var tokenizedData = data
            .Select(text => Tokenizer.Tokenize(vocab, text))
            .Where(t => t.Count >= 2)
            .ToList();

        var rng = new Random(42);
        var indices = new int[tokenizedData.Count];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0.0f;
            int sampleCount = 0;

            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            foreach (var idx in indices)
            {
                var trainingRow = tokenizedData[idx];
                int inputLen = trainingRow.Count - 1;

                var inputArr = new float[inputLen];
                for (int k = 0; k < inputLen; k++)
                    inputArr[k] = trainingRow[k];

                var inputMatrix = Matrix<float>.Build.Dense(
                    1, inputLen, inputArr);

                var input = inputMatrix;
                foreach (var layer in network)
                    input = layer.Forward(input);

                var probs = Softmax(input);

                totalLoss += CrossEntropyLossStep(probs, trainingRow, 1);
                sampleCount++;

                var gradsOutput = ComputeGradientsStep(probs, trainingRow, 1);
                ClipGradients(gradsOutput, 1.0f);

                foreach (var layer in _reversedNetwork)
                    gradsOutput = layer.Backward(gradsOutput, lr);
            }

            if (sampleCount > 0)
                Console.WriteLine(
                    $"Epoch {epoch}: Loss = {totalLoss / sampleCount:F4}");
        }
    }

    private static Matrix<float> Softmax(Matrix<float> logits)
    {
        var result = Matrix<float>.Build.Dense(
            logits.RowCount, logits.ColumnCount);

        for (int i = 0; i < logits.RowCount; i++)
        {
            var row = logits.Row(i);
            var maxVal = row.Maximum();
            var exp = (row - maxVal).PointwiseExp();
            result.SetRow(i, exp / exp.Sum());
        }

        return result;
    }

    private static (int token, float confidence) DecodeWithConfidence(
        Vector<float> probRow)
    {
        var maxIndex = probRow.MaximumIndex();
        return (maxIndex, probRow[maxIndex]);
    }

    private static float CrossEntropyLossStep(
        Matrix<float> probs, List<int> target, int targetOffset)
    {
        float loss = 0.0f;
        int count = probs.RowCount;
        for (int i = 0; i < count; i++)
        {
            float probTarget = probs[i, target[i + targetOffset]];
            loss -= MathF.Log(MathF.Max(1e-15f, probTarget));
        }
        return loss / count;
    }

    private static Matrix<float> ComputeGradientsStep(
        Matrix<float> probs, List<int> target, int targetOffset)
    {
        if (probs.RowCount != target.Count - targetOffset)
            throw new ArgumentException(
                "Probs and target must have the same number of rows");

        var grads = probs.Clone();
        float batchSize = probs.RowCount;

        for (int i = 0; i < grads.RowCount; i++)
            grads[i, target[i + targetOffset]] -= 1.0f;

        return grads / batchSize;
    }

    private static void ClipGradients(Matrix<float> grads, float maxNorm)
    {
        float normSq = 0f;
        for (int i = 0; i < grads.RowCount; i++)
            for (int j = 0; j < grads.ColumnCount; j++)
            {
                float x = grads[i, j];
                normSq += x * x;
            }

        float norm = MathF.Sqrt(normSq);
        if (norm > maxNorm)
        {
            float scale = maxNorm / norm;
            grads.Multiply(scale, grads);
        }
    }
}