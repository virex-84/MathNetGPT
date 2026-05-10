//https://github.com/virex-84

using MessagePack;

namespace Llm.Core;

/// <summary>
/// Состояние модели для сериализации.
/// Содержит все данные, необходимые для сохранения и загрузки.
/// </summary>
[MessagePackObject]
public class ModelState
{
    /// <summary>
    /// Все параметры модели в виде плоского списка матриц.
    /// </summary>
    [Key(0)]
    public List<SerializableMatrix> AllParameters { get; set; }

    /// <summary>
    /// Список слов словаря
    /// </summary>
    [Key(1)]
    public List<string> VocabWords { get; set; }

    /// <summary>
    /// Количество Transformer блоков в модели
    /// </summary>
    [Key(2)]
    public int TransformerBlockCount { get; set; }

    /// <summary>
    /// Размерность embedding
    /// </summary>
    [Key(3)]
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Размерность hidden слоя в FeedForward
    /// </summary>
    [Key(4)]
    public int HiddenDim { get; set; }

    /// <summary>
    /// Максимальная длина последовательности (количество токенов)
    /// </summary>
    [Key(5)]
    public int MaxSeqLen { get; set; }

    public ModelState()
    {
        AllParameters = new List<SerializableMatrix>();
        VocabWords = new List<string>();
        TransformerBlockCount = 1;
        EmbeddingDim = Constants.EMBEDDING_DIM;
        HiddenDim = Constants.HIDDEN_DIM;
        MaxSeqLen = Constants.MAX_SEQ_LEN;
    }
}
